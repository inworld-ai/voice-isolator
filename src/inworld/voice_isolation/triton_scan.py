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
"""The selective scan as one Triton kernel, for NVIDIA GPUs.

The CUDA counterpart of the Metal kernel in ``scan``: each program keeps the
state of a block of one row's channels in registers and walks the frames, so
a scan is one launch however many frames it covers, where the plain
recurrence launches several GPU operations per frame. It takes a carried
state and returns the last, so a stream's few new frames use it as well as a
whole clip. Imported only when a scan runs on CUDA: importing the package
does not import Triton.
"""

import torch
import triton
import triton.language as tl

try:
    # Triton 3; the accurate exp, as the Metal kernel's precise::exp
    from triton.language.extra import libdevice
    _exp = libdevice.exp
except ImportError:  # Triton 2, PyTorch before 2.4
    _exp = tl.math.exp

CHANNELS_PER_PROGRAM = 32


@triton.jit
def _scan_kernel(u, delta, a, b, c, d, z, carried, y, last, frames, channels,
                 states, u_row, u_frame, u_channel, delta_row, delta_frame,
                 delta_channel, b_row, b_frame, b_state, c_row, c_frame,
                 c_state, z_row, z_frame, z_channel, gated: tl.constexpr,
                 carrying: tl.constexpr, block_states: tl.constexpr,
                 block_channels: tl.constexpr):
    row = tl.program_id(0)
    channel = tl.program_id(1) * block_channels + tl.arange(0, block_channels)
    state = tl.arange(0, block_states)
    in_channels = channel < channels
    in_states = state < states
    both = in_channels[:, None] & in_states[None, :]
    rate = tl.load(a + channel[:, None] * states + state[None, :],
                   mask=both,
                   other=0.0)
    skip = tl.load(d + channel, mask=in_channels, other=0.0)
    held = (row * channels + channel[:, None]) * states + state[None, :]
    if carrying:
        h = tl.load(carried + held, mask=both, other=0.0)
    else:
        h = tl.zeros((block_channels, block_states), dtype=tl.float32)
    for t in range(frames):
        step = tl.load(delta + row * delta_row + t * delta_frame +
                       channel * delta_channel,
                       mask=in_channels,
                       other=0.0).to(tl.float32)
        x = tl.load(u + row * u_row + t * u_frame + channel * u_channel,
                    mask=in_channels,
                    other=0.0).to(tl.float32)
        b_t = tl.load(b + row * b_row + t * b_frame + state * b_state,
                      mask=in_states,
                      other=0.0).to(tl.float32)
        c_t = tl.load(c + row * c_row + t * c_frame + state * c_state,
                      mask=in_states,
                      other=0.0).to(tl.float32)
        h = _exp(step[:, None] * rate) * h + (step * x)[:, None] * b_t[None, :]
        out = tl.sum(h * c_t[None, :], axis=1) + x * skip
        if gated:
            g = tl.load(z + row * z_row + t * z_frame + channel * z_channel,
                        mask=in_channels,
                        other=0.0).to(tl.float32)
            out = out * (g / (1.0 + _exp(-g)))
        tl.store(y + (row * frames + t) * channels + channel,
                 out,
                 mask=in_channels)
    tl.store(last + held, h, mask=both)


def selective_scan(
    u: torch.Tensor,
    delta: torch.Tensor,
    a: torch.Tensor,
    b: torch.Tensor,
    c: torch.Tensor,
    d: torch.Tensor,
    z: torch.Tensor | None = None,
    state: torch.Tensor | None = None,
) -> tuple[torch.Tensor, torch.Tensor]:
    """``scan.selective_scan`` on CUDA tensors, in one kernel launch.

    The inputs may be strided views in any floating dtype; each is read in
    place and computed in float32. Arguments and results are those of
    ``scan.selective_scan``.
    """
    batch, frames, channels = u.shape
    states = a.shape[-1]
    y = u.new_empty(batch, frames, channels, dtype=torch.float32)
    last = u.new_empty(batch, channels, states, dtype=torch.float32)
    if not batch * channels:
        return y, last
    carried = None if state is None else state.float().contiguous()
    gate = u if z is None else z
    grid = (batch, triton.cdiv(channels, CHANNELS_PER_PROGRAM))
    _scan_kernel[grid](u,
                       delta,
                       a.float().contiguous(),
                       b,
                       c,
                       d.float().contiguous(),
                       gate,
                       last if carried is None else carried,
                       y,
                       last,
                       frames,
                       channels,
                       states,
                       *u.stride(),
                       *delta.stride(),
                       *b.stride(),
                       *c.stride(),
                       *gate.stride(),
                       gated=z is not None,
                       carrying=carried is not None,
                       block_states=triton.next_power_of_2(states),
                       block_channels=CHANNELS_PER_PROGRAM)
    return y, last
