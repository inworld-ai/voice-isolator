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
"""The graphed stream against the plain one."""

# pylint: disable=protected-access
import pytest
import torch

from inworld.voice_isolation import cuda_graphs
from inworld.voice_isolation import streaming

SR = 48000
WINDOWS = dict(window_frames=40, crossfade_frames=10)


def _cuda_or_skip():
    if not torch.cuda.is_available():
        pytest.skip("CUDA graphs run on CUDA")
    pytest.importorskip("triton.language.extra.libdevice")


def _clip(seconds, seed=0):
    g = torch.Generator().manual_seed(seed)
    return torch.randn(int(seconds * SR), generator=g) * 0.3


def _run(stream, clip, sizes):
    pieces, pos, index = [], 0, 0
    while pos < clip.shape[-1]:
        step = sizes[index % len(sizes)]
        pieces.append(stream.push(clip[..., pos:pos + step]))
        pos += step
        index += 1
    pieces.append(stream.flush())
    return torch.cat(pieces, dim=-1)


@pytest.mark.parametrize("autocast", [None, torch.bfloat16])
@pytest.mark.parametrize("batch", [1, 3])
@pytest.mark.parametrize("sizes", [(64,), (37, 101, 5)])
def test_the_graphed_stream_equals_the_plain_one(tiny, autocast, batch, sizes):
    _cuda_or_skip()
    model = tiny.dual(width=64, num_heads=2).cuda()
    model.set_lookahead(2)
    clip = torch.stack([_clip(0.1, seed) for seed in range(batch)])
    kw = dict(block_frames=4, autocast_dtype=autocast, **WINDOWS)

    graphed = cuda_graphs.GraphedStream(model, **kw)
    out = _run(graphed, clip, sizes)

    plain = _run(streaming.Stream(model, **kw), clip, sizes)
    assert graphed._graphs, "no block was replayed as a graph"
    assert len(graphed.boundaries) > 2
    assert out.shape == plain.shape == clip.shape
    assert torch.equal(out, plain)


@pytest.mark.parametrize("kw", [{}, dict(cache_frames=30, **WINDOWS)])
def test_a_stream_a_graph_cannot_replay_is_refused(tiny, kw):
    _cuda_or_skip()
    model = tiny.dual(width=64, num_heads=2).cuda()
    model.set_lookahead(2)

    with pytest.raises(ValueError):
        cuda_graphs.GraphedStream(model, **kw)


def test_a_cpu_stream_is_refused(tiny):
    model = tiny.dual()
    model.set_lookahead(2)

    with pytest.raises(ValueError, match="CUDA device"):
        cuda_graphs.GraphedStream(model, **WINDOWS)
