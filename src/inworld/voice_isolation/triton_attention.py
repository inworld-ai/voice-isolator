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
"""A stream's time attention over its cache as one Triton kernel.

Each new block of a stream rotates its queries and keys, writes the keys and
values into the lane's cache, attends over the cached past and itself, and,
for the released checkpoints, removes each output's component along its own
value: a dozen GPU operations per attention block, each over a few frames.
Here one program per (row, head) does all of it. It reads only the cache
tiles a query may see, keeps the softmax online over them and runs both
products on the tensor cores, and writes the output in the layout the
projection reads. The keys it writes equal ``modeling.apply_rope``'s in the
attention's dtype; the output equals the eager attention's to float32
rounding (or within one bfloat16 rounding of it). Needs Triton 3; imported
only when a CUDA stream attends.
"""

import torch
import triton
import triton.language as tl
from triton.language.extra import libdevice

FRAMES_PER_PROGRAM = 16
KEYS_PER_TILE = 64


@triton.jit
def _rotate(a, b, c, d, sign: tl.constexpr, dtype: tl.constexpr):
    """a b + sign c d as ``apply_rope`` computes it in ``dtype``: each product
    and the sum rounded to it, never fused into one multiply-add."""
    first = libdevice.mul_rn(a, b).to(dtype).to(tl.float32)
    second = libdevice.mul_rn(c, d).to(dtype).to(tl.float32)
    return libdevice.add_rn(first, sign * second).to(dtype)


@triton.jit
def _dot(a, b, ieee: tl.constexpr):
    if ieee:
        return tl.dot(a, b, input_precision="ieee")
    return tl.dot(a, b)


@triton.jit
def _attend_kernel(qkv, keys, values, cos, sin, out, first_at, start, frames,
                   window, scale, qkv_row, qkv_frame, qkv_part, qkv_head,
                   cache_row, cache_head, cache_slot, out_row, out_frame,
                   first_in_memory: tl.constexpr, exclusive: tl.constexpr,
                   ieee: tl.constexpr, head_dim: tl.constexpr,
                   block_frames: tl.constexpr, block_keys: tl.constexpr):
    half: tl.constexpr = head_dim // 2
    if first_in_memory:
        first = tl.load(first_at).to(tl.int32)
    else:
        first = first_at
    row = tl.program_id(0)
    head = tl.program_id(1)
    frame = tl.arange(0, block_frames)
    new = frame < frames
    lane = tl.arange(0, half)
    dim = tl.arange(0, head_dim)
    slot = first + frame
    table = slot[:, None] * half + lane[None, :]
    base = qkv + row * qkv_row + frame[:, None] * qkv_frame + head * qkv_head
    q1 = tl.load(base + lane[None, :], mask=new[:, None], other=0.0)
    dtype: tl.constexpr = q1.dtype
    cos_t = tl.load(cos + table, mask=new[:, None],
                    other=0.0).to(dtype).to(tl.float32)
    sin_t = tl.load(sin + table, mask=new[:, None],
                    other=0.0).to(dtype).to(tl.float32)
    q1 = q1.to(tl.float32)
    q2 = tl.load(base + half + lane[None, :], mask=new[:, None],
                 other=0.0).to(tl.float32)
    k1 = tl.load(base + qkv_part + lane[None, :], mask=new[:, None],
                 other=0.0).to(tl.float32)
    k2 = tl.load(base + qkv_part + half + lane[None, :],
                 mask=new[:, None],
                 other=0.0).to(tl.float32)
    v = tl.load(base + 2 * qkv_part + dim[None, :],
                mask=new[:, None],
                other=0.0)
    q1, q2 = (_rotate(q1, cos_t, q2, sin_t, -1.0,
                      dtype), _rotate(q1, sin_t, q2, cos_t, 1.0, dtype))
    k1, k2 = (_rotate(k1, cos_t, k2, sin_t, -1.0,
                      dtype), _rotate(k1, sin_t, k2, cos_t, 1.0, dtype))
    cached = row * cache_row + head * cache_head
    written = cached + slot[:, None] * cache_slot
    tl.store(keys + written + lane[None, :], k1, mask=new[:, None])
    tl.store(keys + written + half + lane[None, :], k2, mask=new[:, None])
    tl.store(values + written + dim[None, :], v, mask=new[:, None])

    peak = tl.full([block_frames], -1e30, dtype=tl.float32)
    total = tl.zeros([block_frames], dtype=tl.float32)
    acc = tl.zeros([block_frames, head_dim], dtype=tl.float32)
    # the cached past [start, first), from the first tile any query may see
    lowest = tl.maximum(start, first - window + 1)
    for tile in range(lowest - lowest % block_keys, first, block_keys):
        key = tile + tl.arange(0, block_keys)
        held = (key >= start) & (key < first)
        k1_t = tl.load(keys + cached + key[None, :] * cache_slot +
                       lane[:, None],
                       mask=held[None, :],
                       other=0.0)
        k2_t = tl.load(keys + cached + key[None, :] * cache_slot + half +
                       lane[:, None],
                       mask=held[None, :],
                       other=0.0)
        score = _dot(q1, k1_t, ieee) + _dot(q2, k2_t, ieee)
        seen = held[None, :] & (key[None, :] > slot[:, None] - window)
        score = tl.where(seen, score * scale, float("-inf"))
        top = tl.maximum(peak, tl.max(score, 1))
        decay = tl.exp(peak - top)
        p = tl.exp(score - top[:, None])
        total = total * decay + tl.sum(p, 1)
        v_t = tl.load(values + cached + key[:, None] * cache_slot +
                      dim[None, :],
                      mask=held[:, None],
                      other=0.0)
        acc = acc * decay[:, None] + _dot(p.to(dtype), v_t, ieee)
        peak = top
    # the block's own keys, from registers: frame j is seen by frame i <= j
    score = _dot(q1, tl.trans(k1), ieee) + _dot(q2, tl.trans(k2), ieee)
    seen = ((frame[None, :] <= frame[:, None]) & new[None, :] &
            (slot[None, :] > slot[:, None] - window))
    score = tl.where(seen, score * scale, float("-inf"))
    top = tl.maximum(peak, tl.max(score, 1))
    decay = tl.exp(peak - top)
    p = tl.exp(score - top[:, None])
    total = total * decay + tl.sum(p, 1)
    acc = acc * decay[:, None] + _dot(p.to(dtype), v, ieee)
    output = acc / total[:, None]
    if exclusive:
        # rounded to the attention's dtype first, as its output would be
        output = output.to(dtype).to(tl.float32)
        own = v.to(tl.float32)
        along = tl.sum(output * own, 1) / tl.maximum(tl.sum(own * own, 1), 1e-6)
        output = output - along[:, None] * own
    tl.store(out + row * out_row + frame[:, None] * out_frame +
             head * head_dim + dim[None, :],
             output.to(dtype),
             mask=new[:, None])


def supports(head_dim: int, frames: int) -> bool:
    """Whether the kernel takes this head size and block of frames."""
    return (head_dim >= 32 and head_dim & (head_dim - 1) == 0 and
            frames <= FRAMES_PER_PROGRAM)


def attend(qkv: torch.Tensor, keys: torch.Tensor, values: torch.Tensor,
           cos: torch.Tensor, sin: torch.Tensor, first: int | torch.Tensor,
           start: int, window: int, exclusive: bool, out: torch.Tensor) -> None:
    """One lane's attention block over its cache.

    Args:
        qkv: The block's projections [rows, frames, 3, heads, head_dim], the
            last dimension contiguous.
        keys: The lane's key buffer [rows, heads, slots, head_dim]; the
            block's rotated keys are written at slots ``first`` onward.
        values: The lane's value buffer, like ``keys``.
        cos: Rotary cosines [slots, head_dim / 2] in float32, by slot.
        sin: Rotary sines, like ``cos``.
        first: The slot of the block's first frame, or a 0-d integer tensor
            on the device holding it: the kernel then reads it, so a CUDA
            graph replays the block at the lane's current length.
        start: The first cached slot the block may see.
        window: The trained attention window in frames, 0 for none.
        exclusive: Whether to remove each output's component along its own
            value.
        out: Receives the output [rows, frames, heads * head_dim].
    """
    rows, frames, _, heads, head_dim = qkv.shape
    launch = _attend_kernel[(rows, heads)]
    launch(qkv,
           keys,
           values,
           cos,
           sin,
           out,
           first,
           start,
           frames,
           window or 1 << 30,
           head_dim**-0.5,
           qkv.stride(0),
           qkv.stride(1),
           qkv.stride(2),
           qkv.stride(3),
           keys.stride(0),
           keys.stride(1),
           keys.stride(2),
           out.stride(0),
           out.stride(1),
           first_in_memory=isinstance(first, torch.Tensor),
           exclusive=exclusive,
           ieee=qkv.dtype == torch.float32,
           head_dim=head_dim,
           block_frames=FRAMES_PER_PROGRAM,
           block_keys=KEYS_PER_TILE)
