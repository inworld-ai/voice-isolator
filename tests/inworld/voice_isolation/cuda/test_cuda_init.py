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
"""The kernel loader of ``inworld.voice_isolation.cuda``."""

import subprocess
import sys

import torch

from inworld.voice_isolation import cuda


def test_importing_the_package_imports_no_triton():
    code = ("import sys; from inworld import voice_isolation; "
            "print('triton' in sys.modules)")

    out = subprocess.run([sys.executable, "-c", code],
                         capture_output=True,
                         text=True,
                         check=True)

    assert out.stdout.strip() == "False"


def test_a_kernel_without_triton_is_none(monkeypatch):
    monkeypatch.setitem(sys.modules, "triton", None)
    monkeypatch.setattr(cuda, "_KERNELS", {})
    for name in ("scan_kernel", "attention_kernel"):
        monkeypatch.delitem(sys.modules, f"{cuda.__name__}.{name}", False)

    assert cuda.kernel("scan_kernel") is None
    assert cuda.kernel("attention_kernel") is None


def test_rocm_takes_the_eager_code(monkeypatch):
    monkeypatch.setattr(torch.version, "hip", "6.2")

    assert cuda.kernel("scan_kernel") is None
