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
"""Faster scans and streams on NVIDIA GPUs, with Triton and CUDA graphs.

On CUDA, a stream's block of a few frames is several hundred small GPU
operations, and launching them takes longer than running them. This package
cuts the launches:

- ``scan_kernel``: the selective scan as one Triton kernel. ``scan`` runs
  every CUDA scan through it, a stream's carried one included.
- ``attention_kernel``: a stream's time attention block as one Triton kernel
  (Triton 3). ``streaming`` attends through it on CUDA.
- ``graphs``: ``GraphedStream``, a windowed stream whose blocks replay as
  CUDA graphs: ``VoiceIsolator.stream(cuda_graphs=True)``.

Triton comes with PyTorch's CUDA builds on Linux. Where it is missing, or
too old for a kernel, or PyTorch is built for ROCm, ``kernel`` returns None
and the eager code runs.
Importing this package imports neither Triton nor the kernels.
"""

import importlib
import types

import torch

_KERNELS: dict[str, types.ModuleType | None] = {}


def kernel(name: str) -> types.ModuleType | None:
    """The kernel module ``name`` (``"scan_kernel"`` or
    ``"attention_kernel"``), imported on first use, or None where its Triton
    is missing (or on ROCm, where they are untested)."""
    if torch.version.hip is not None:
        return None
    if name not in _KERNELS:
        try:
            _KERNELS[name] = importlib.import_module(f"{__name__}.{name}")
        except ImportError:
            _KERNELS[name] = None
    return _KERNELS[name]
