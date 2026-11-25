# Copyright 2026 Inworld AI
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
"""The fused stream attention against the eager one, on CUDA."""

import pytest
import torch

from inworld.voice_isolation import cuda
from inworld.voice_isolation import model as modeling
from inworld.voice_isolation import streaming

SR = 48000


def _fused_or_skip():
    if not torch.cuda.is_available():
        pytest.skip("the fused attention runs on CUDA")
    pytest.importorskip("triton.language.extra.libdevice")
    return cuda.kernel("attention_kernel")


def _eager(qkv, keys, values, cos, sin, first, start, window, exclusive):
    """The eager attention block: rotate, write, attend, remove."""
    frames = qkv.shape[1]
    query, key, value = (t.transpose(1, 2) for t in qkv.unbind(2))
    cos_t = cos[first:first + frames].to(qkv.dtype)
    sin_t = sin[first:first + frames].to(qkv.dtype)
    keys[:, :, first:first + frames] = modeling.apply_rope(key, cos_t, sin_t)
    values[:, :, first:first + frames] = value
    slots = torch.arange(start, first + frames, device=qkv.device)
    seen = slots[None, :] <= slots[-frames:, None]
    if window:
        seen &= slots[None, :] > slots[-frames:, None] - window
    output = torch.nn.functional.scaled_dot_product_attention(
        modeling.apply_rope(query, cos_t, sin_t),
        keys[:, :, start:first + frames],
        values[:, :, start:first + frames],
        attn_mask=seen)
    if exclusive:
        output = modeling.exclusive(output, value)
    return output.transpose(1, 2).reshape(qkv.shape[0], frames, -1)


@pytest.mark.parametrize("first,start,frames,window", [(0, 0, 2, 0),
                                                       (3, 0, 1, 0),
                                                       (130, 0, 4, 0),
                                                       (200, 70, 7, 0),
                                                       (300, 0, 16, 128),
                                                       (63, 1, 2, 64)])
@pytest.mark.parametrize("dtype,head_dim", [(torch.float32, 32),
                                            (torch.float32, 64),
                                            (torch.float16, 64),
                                            (torch.bfloat16, 64),
                                            (torch.bfloat16, 128)])
@pytest.mark.parametrize("exclusive", [False, True])
def test_the_fused_block_equals_the_eager_one(first, start, frames, window,
                                              dtype, head_dim, exclusive):
    attention_kernel = _fused_or_skip()
    g = torch.Generator().manual_seed(first + frames)
    rows, heads, slots = 5, 2, 400
    qkv = torch.randn(rows, frames, 3, heads, head_dim,
                      generator=g).cuda().to(dtype)
    keys = torch.randn(rows, heads, slots, head_dim,
                       generator=g).cuda().to(dtype)
    values = torch.randn(rows, heads, slots, head_dim,
                         generator=g).cuda().to(dtype)
    cos, sin = modeling.rope_tables(slots, head_dim, "cuda", torch.float32)
    eager_keys, eager_values = keys.clone(), values.clone()
    out = torch.empty(rows,
                      frames,
                      heads * head_dim,
                      device="cuda",
                      dtype=dtype)

    attention_kernel.attend(qkv, keys, values, cos, sin, first, start, window,
                            exclusive, out)

    ref = _eager(qkv, eager_keys, eager_values, cos, sin, first, start, window,
                 exclusive)
    assert torch.equal(keys, eager_keys)
    assert torch.equal(values, eager_values)
    if dtype == torch.float32:
        assert torch.allclose(out, ref, atol=1e-5, rtol=1e-5)
    else:
        assert (out.float() - ref.float()).abs().max() <= 2e-2 * ref.abs().max()


@pytest.mark.parametrize("head_dim,frames,dtype,taken", [
    (64, 16, torch.bfloat16, True),
    (128, 1, torch.float16, True),
    (64, 4, torch.float32, True),
    (128, 4, torch.float32, False),
    (256, 4, torch.bfloat16, False),
    (16, 4, torch.bfloat16, False),
    (96, 4, torch.bfloat16, False),
    (64, 17, torch.bfloat16, False),
    (64, 4, torch.float64, False),
])
def test_the_kernel_takes_the_blocks_it_fits(head_dim, frames, dtype, taken):
    attention_kernel = _fused_or_skip()
    if torch.cuda.get_device_capability() < (8, 0):
        pytest.skip("the fused attention needs compute capability 8.0")

    assert attention_kernel.supports(head_dim, frames, dtype,
                                     torch.device("cuda")) == taken


def _clip(seconds, seed=0):
    g = torch.Generator().manual_seed(seed)
    return torch.randn(int(seconds * SR), generator=g) * 0.3


def _run(model, clip, sizes, **kw):
    stream = streaming.Stream(model, **kw)
    pieces, pos, index = [], 0, 0
    while pos < clip.shape[-1]:
        step = sizes[index % len(sizes)]
        pieces.append(stream.push(clip[..., pos:pos + step]))
        pos += step
        index += 1
    pieces.append(stream.flush())
    return torch.cat(pieces, dim=-1)


@pytest.mark.parametrize("build,kw", [
    ("dual", dict(block_frames=4)),
    ("dual", dict(block_frames=7, cache_frames=12)),
    ("dual", dict(block_frames=4, window_frames=40, crossfade_frames=10)),
    ("windowed", dict(block_frames=1)),
    ("windowed", dict(block_frames=16, cache_frames=9)),
])
@pytest.mark.parametrize("batch", [1, 3])
def test_a_cuda_stream_attends_as_the_eager_one(tiny, monkeypatch, build, kw,
                                                batch):
    _fused_or_skip()
    model = getattr(tiny, build)(width=64, num_heads=2).cuda()
    model.set_lookahead(2)
    clip = _clip(0.1)
    if batch > 1:
        clip = torch.stack([_clip(0.1, seed) for seed in range(batch)])

    fused = _run(model, clip, (37, 101, 5), **kw)
    kernel = cuda.kernel
    monkeypatch.setattr(
        cuda, "kernel", lambda name: None
        if name == "attention_kernel" else kernel(name))
    eager = _run(model, clip, (37, 101, 5), **kw)

    assert fused.shape == eager.shape == clip.shape
    assert torch.allclose(fused, eager, atol=1e-5)


@pytest.mark.parametrize("autocast,taken", [(torch.bfloat16, True),
                                            (None, False)])
def test_a_stream_with_heads_of_128_attends_as_the_eager_one(
        tiny, monkeypatch, autocast, taken):
    attention_kernel = _fused_or_skip()
    if torch.cuda.get_device_capability() < (8, 0):
        pytest.skip("the fused attention needs compute capability 8.0")
    model = tiny.dual(width=128, num_heads=1).cuda()
    model.set_lookahead(2)
    clip = _clip(0.1)
    kw = dict(block_frames=4, autocast_dtype=autocast)
    calls = []
    attend = attention_kernel.attend
    monkeypatch.setattr(attention_kernel, "attend",
                        lambda *args: calls.append(1) or attend(*args))

    fused = _run(model, clip, (37, 101, 5), **kw)
    kernel = cuda.kernel
    monkeypatch.setattr(
        cuda, "kernel", lambda name: None
        if name == "attention_kernel" else kernel(name))
    eager = _run(model, clip, (37, 101, 5), **kw)

    # float32 tiles of 128 do not fit: that stream attends eagerly
    assert bool(calls) == taken
    assert fused.shape == eager.shape == clip.shape
    if taken:
        assert (fused - eager).abs().max() <= 1e-2 * eager.abs().max()
    else:
        assert torch.equal(fused, eager)
