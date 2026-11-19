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
"""The Triton scan against the frame-by-frame recurrence, on CUDA."""

import pytest
import torch

from inworld.voice_isolation import scan


def _triton_scan_or_skip():
    if not torch.cuda.is_available():
        pytest.skip("the Triton scan runs on CUDA")
    pytest.importorskip("triton")
    # pylint: disable-next=import-outside-toplevel
    from inworld.voice_isolation import triton_scan
    return triton_scan


def _inputs(batch, frames, channels, state, seed=0):
    g = torch.Generator().manual_seed(seed)

    def rand(*shape):
        return torch.randn(*shape, generator=g)

    return dict(
        u=rand(batch, frames, channels),
        delta=torch.nn.functional.softplus(rand(batch, frames, channels) * 2),
        a=-torch.exp(rand(channels, state)),
        b=rand(batch, frames, state),
        c=rand(batch, frames, state),
        d=rand(channels),
        z=rand(batch, frames, channels),
    )


def _cuda(kw):
    return {k: None if v is None else v.cuda() for k, v in kw.items()}


@pytest.mark.parametrize("frames", [1, 2, 46, 203])
@pytest.mark.parametrize("carried", [False, True])
@pytest.mark.parametrize("gated", [False, True])
@pytest.mark.parametrize("states", [3, 16])
def test_the_triton_scan_matches_the_recurrence(frames, carried, gated, states):
    triton_scan = _triton_scan_or_skip()
    kw = _inputs(3, frames, 37, states, seed=frames)
    if not gated:
        kw["z"] = None
    state = torch.randn(3, 37, states) if carried else None

    y, h = triton_scan.selective_scan(**_cuda(kw),
                                      state=None
                                      if state is None else state.cuda())

    ref, href = scan.selective_scan_reference(**kw, state=state)
    assert y.is_cuda and y.dtype == torch.float32 and y.shape == ref.shape
    assert torch.allclose(y.cpu(), ref, atol=1e-5, rtol=1e-5)
    assert torch.allclose(h.cpu(), href, atol=1e-5, rtol=1e-5)


def test_a_state_count_off_a_power_of_two_is_masked():
    triton_scan = _triton_scan_or_skip()
    kw = _inputs(3, 46, 37, 17, seed=7)
    state = torch.randn(3, 37, 17)

    y, h = triton_scan.selective_scan(**_cuda(kw), state=state.cuda())

    ref, href = scan.selective_scan_reference(**kw, state=state)
    assert torch.allclose(y.cpu(), ref, atol=1e-5, rtol=1e-5)
    assert torch.allclose(h.cpu(), href, atol=1e-5, rtol=1e-5)


def test_the_triton_scan_reads_strided_half_precision_inputs():
    triton_scan = _triton_scan_or_skip()
    kw = _cuda(_inputs(2, 9, 40, 16, seed=3))
    packed = torch.cat((kw["b"], kw["c"]), -1).bfloat16()
    views = dict(kw,
                 u=kw["u"].bfloat16().transpose(1,
                                                2).contiguous().transpose(1, 2),
                 b=packed[..., :16],
                 c=packed[..., 16:],
                 z=kw["z"].bfloat16())

    y, h = triton_scan.selective_scan(**views)

    ref, href = scan.selective_scan_reference(**views)
    assert torch.allclose(y, ref, atol=1e-5, rtol=1e-5)
    assert torch.allclose(h, href, atol=1e-5, rtol=1e-5)


def test_a_cuda_stream_splits_the_scan_anywhere(monkeypatch):
    _triton_scan_or_skip()
    monkeypatch.setattr(scan, "kernel", lambda: None)
    kw = _cuda(_inputs(2, 40, 6, 3, seed=1))
    whole, final = scan.selective_scan(**kw)

    parts, state = [], None
    for start, stop in ((0, 1), (1, 17), (17, 18), (18, 40)):
        piece = {
            k: (v[:, start:stop] if v.ndim == 3 else v) for k, v in kw.items()
        }
        y, state = scan.selective_scan(**piece, state=state)
        parts.append(y)

    ref, href = scan.selective_scan_reference(**kw)
    assert torch.allclose(torch.cat(parts, 1), whole, atol=1e-5)
    assert torch.allclose(state, final, atol=1e-5)
    assert torch.allclose(whole, ref, atol=1e-5, rtol=1e-5)
    assert torch.allclose(final, href, atol=1e-5, rtol=1e-5)


def test_an_empty_batch_scans_to_nothing():
    triton_scan = _triton_scan_or_skip()
    kw = _cuda(_inputs(0, 5, 8, 4))

    y, h = triton_scan.selective_scan(**kw)

    assert y.shape == (0, 5, 8) and h.shape == (0, 8, 4)


def test_more_states_than_the_triton_scan_holds_take_the_torch_scan(
        monkeypatch):
    _triton_scan_or_skip()
    monkeypatch.setattr(scan, "kernel", lambda: None)
    kw = _inputs(2, 9, 5, scan.TRITON_STATES + 1, seed=5)

    y, _ = scan.selective_scan(**_cuda(kw))

    ref, _ = scan.selective_scan_reference(**kw)
    assert torch.allclose(y.cpu(), ref, atol=1e-5, rtol=1e-5)
