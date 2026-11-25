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
"""A windowed stream whose blocks replay as CUDA graphs.

A stream's block of a few frames runs several hundred small GPU operations;
on a fast GPU, launching them takes longer than running them. A CUDA graph
records a block's operations once and replays them in a single launch. It
replays them on the same memory, so here a lane's state stays at fixed
addresses: its attention keys and values in buffers of the lane's full
length, which a windowed lane never outgrows, with the lane's length on the
GPU, read by the fused attention kernel (``attention_kernel``) and advanced
in place; and its Mamba state in buffers that every block's new state is
copied into. A block of a size seen before replays its graph (one per block
size and per one or two lanes); a new size runs once directly first, so a
one-off block, a flush's for one, is never recorded.

The output equals ``streaming.Stream``'s with the same arguments: the
stream's arithmetic is unchanged, and the empty slots attention now skips
contribute nothing. Windowed streams on CUDA only, with Triton 3 and the
fused attention's head sizes; the plain stream compacts its cache, which a
graph cannot replay.
"""

import contextlib

import torch

from inworld.voice_isolation import cuda
from inworld.voice_isolation import model as modeling
from inworld.voice_isolation import streaming


class _Attention:
    """A graphed lane's attention past: fixed buffers, length on the GPU."""

    def __init__(self, rows, heads, slots, head_dim, dtype, device):
        shape = (rows, heads, slots, head_dim)
        self.keys = torch.zeros(shape, dtype=dtype, device=device)
        self.values = torch.zeros(shape, dtype=dtype, device=device)
        self.end = torch.zeros((), dtype=torch.long, device=device)

    def claim(self, cache, key, value) -> tuple[torch.Tensor, int]:
        """``streaming._AttentionState.claim``: the lane's buffers already
        hold its window, so the new keys go at its length on the device, and
        every slot before them is its own."""
        del cache, key, value
        return self.end, 0


class _Mamba:
    """A graphed lane's Mamba state: fixed buffers, zeros before the first
    frame, that a new state is copied into."""

    def __init__(self, rows, inner, convolution, states, dtype, device):
        self._parts = (torch.zeros(rows,
                                   inner,
                                   convolution - 1,
                                   dtype=dtype,
                                   device=device),
                       torch.zeros(rows, inner, states, device=device))

    @property
    def carried(self) -> tuple[torch.Tensor, torch.Tensor]:
        return self._parts

    @carried.setter
    def carried(self, parts) -> None:
        for buffer, part in zip(self._parts, parts):
            buffer.copy_(part)


class GraphedStream(streaming.Stream):
    """``streaming.Stream`` with every repeated block replayed as a graph.

    Takes the arguments of ``streaming.Stream``; needs a CUDA model, windowed
    attention (``window_frames`` and ``crossfade_frames``) whose lanes fit
    the cache, Triton 3 and a head size the fused attention takes. Each
    stream records its own graphs, the first time a block size repeats.

    Raises:
        ValueError: When the stream cannot be graphed.
    """

    def __init__(self, *args, **kwargs) -> None:
        super().__init__(*args, **kwargs)
        config = self.model.config
        if self._audio.device.type != "cuda":
            raise ValueError("CUDA graphs need the model on a CUDA device.")
        if self.window_frames is None:
            raise ValueError("CUDA graphs need windowed attention:"
                             " window_frames and crossfade_frames.")
        lane = self.window_frames + self.crossfade_frames
        if lane > self.cache_frames:
            raise ValueError(
                f"CUDA graphs need a window and crossfade ({lane} frames)"
                f" within the attention cache ({self.cache_frames}).")
        fused = cuda.kernel("attention_kernel")
        if fused is None or not fused.supports(
                config.width // config.num_heads, self.block_frames,
                self.autocast_dtype or torch.float32, self._audio.device):
            raise ValueError(
                "CUDA graphs need Triton 3, a GPU of compute capability 8.0"
                " or later, blocks of up to 16 frames and a head size the"
                " fused attention takes (see attention_kernel.supports).")
        self._lanes = None
        self._lane_a = 0
        self._crossfading = False
        self._graphs = {}
        self._seen = set()
        self._pool = None

    def _allocate(self, batch: int) -> None:
        config = self.model.config
        device = self._audio.device
        dtype = self.autocast_dtype or torch.float32
        rows = batch * len(self.model.edges)
        heads = config.num_heads
        lanes = []
        for _ in range(2):
            lanes.append([
                _Attention(rows, heads, self._attention.capacity,
                           config.width // heads, dtype, device)
                if isinstance(block, modeling.DualPathAttention) else _Mamba(
                    rows, config.width * config.mamba_expand, config.mamba_conv,
                    config.mamba_state, dtype, device)
                for block in self.model.blocks
            ])
        self._lanes = lanes

    def _autocast(self):
        # recorded casts must be replayed, not served from autocast's cache
        if self.autocast_dtype is None:
            return contextlib.nullcontext()
        return torch.autocast(device_type="cuda",
                              dtype=self.autocast_dtype,
                              cache_enabled=False)

    def _step(self, spectra: torch.Tensor, lanes: tuple) -> torch.Tensor:
        with self._autocast():
            features = self._bands.encode(self.model, spectra.permute(1, 2, 0))
            film = self.model.film()
            return self._bands.decode(
                self._backbone(features, [self._lanes[i] for i in lanes], film))

    def _state(self) -> list:
        return [
            part for lane in self._lanes for state in lane for part in ((
                state.end,) if isinstance(state, _Attention) else state.carried)
        ]

    def _record(self, spectra: torch.Tensor, lanes: tuple):
        """Records a block's graph. Warming up runs the block, so the lanes'
        state is restored after it; the slots it wrote are past every lane's
        length and are written again before they are read."""
        saved = [part.clone() for part in self._state()]
        static = spectra.clone()
        side = torch.cuda.Stream()
        side.wait_stream(torch.cuda.current_stream())
        with torch.cuda.stream(side):
            self._step(static, lanes)
        torch.cuda.current_stream().wait_stream(side)
        for part, value in zip(self._state(), saved):
            part.copy_(value)
        if self._pool is None:
            self._pool = torch.cuda.graph_pool_handle()
        graph = torch.cuda.CUDAGraph()
        with torch.cuda.graph(graph, pool=self._pool):
            output = self._step(static, lanes)
        return graph, static, output

    def _replay(self, spectra: torch.Tensor, lanes: tuple) -> torch.Tensor:
        key = (spectra.shape[0], lanes)
        # graphs, their side stream and Triton use the current device
        with torch.cuda.device(spectra.device):
            if key not in self._graphs:
                if key not in self._seen:
                    self._seen.add(key)
                    return self._step(spectra, lanes)
                self._graphs[key] = self._record(spectra, lanes)
            graph, static, output = self._graphs[key]
            static.copy_(spectra)
            graph.replay()
            return output.clone()

    def _run_block(self, spectra: torch.Tensor) -> torch.Tensor:
        """``Stream._run_block`` and ``Stream._windowed`` over the graphed
        lanes: lane B starts with lane A's Mamba state and an empty past, the
        two blend over the crossfade and lane B carries on from the boundary.
        """
        if self._lanes is None:
            self._allocate(spectra.shape[1])
        first = self._frames % self.window_frames
        crossfade_start = self.window_frames - self.crossfade_frames
        a, b = self._lane_a, 1 - self._lane_a
        if first >= crossfade_start and not self._crossfading:
            for state_a, state_b in zip(self._lanes[a], self._lanes[b]):
                if isinstance(state_b, _Attention):
                    state_b.end.zero_()
                else:
                    state_b.carried = state_a.carried
            self._crossfading = True
        if not self._crossfading:
            return self._replay(spectra, (a,)).permute(2, 0, 1, 3)
        raw = self._crossfade(self._replay(spectra, (a, b)), first)
        frames = spectra.shape[0]
        if first + frames == self.window_frames:
            self.boundaries.append(self._frames + frames)
            self._lane_a, self._crossfading = b, False
        return raw.permute(2, 0, 1, 3)
