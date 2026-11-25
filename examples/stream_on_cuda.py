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
"""Streams a recording on an NVIDIA GPU, with and without CUDA graphs.

Plays the recording into a live session in 20 ms pieces, as a microphone
delivers them, once as a plain stream and once with ``cuda_graphs=True``,
and reports how long each push took and how far the two outputs differ.
``--streams`` runs that many copies in lockstep as one batch, as a server
with several live calls would. The pushes run back to back rather than
every 20 ms, so the run also shows how much faster than real time it went.

    python examples/stream_on_cuda.py demo/examples/sample_14.wav
    python examples/stream_on_cuda.py call.wav --streams 8 --output clean.wav
"""

import argparse
import time

import numpy as np
import torch

from inworld import voice_isolation

PIECE = 960  # 20 ms at 48 kHz


def stream(isolator: voice_isolation.VoiceIsolator, clips: torch.Tensor,
           cuda_graphs: bool) -> tuple[np.ndarray, list[float]]:
    """The voice of ``clips`` [streams, samples] and each push's seconds."""
    session = isolator.stream(cuda_graphs=cuda_graphs)
    voice, seconds = [], []
    for start in range(0, clips.shape[-1], PIECE):
        began = time.perf_counter()
        voice.append(session.push(clips[:, start:start + PIECE].numpy()))
        seconds.append(time.perf_counter() - began)
    voice.append(session.flush(as_numpy=True))
    return np.concatenate(voice, axis=-1), seconds


def report(name: str, seconds: list[float], audio_seconds: float) -> None:
    # the first pushes compile kernels and record graphs
    steady = np.array(seconds[len(seconds) // 10:]) * 1e3
    print(f"{name:12s} push p50 {np.percentile(steady, 50):5.2f} ms,"
          f" p95 {np.percentile(steady, 95):5.2f} ms;"
          f" {audio_seconds / sum(seconds):5.1f}x real time")


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=__doc__.split("\n", maxsplit=1)[0])
    parser.add_argument("recording")
    parser.add_argument("--model", default="inworld/voice-isolator")
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--streams",
                        type=int,
                        default=1,
                        help="copies of the recording streamed as one batch")
    parser.add_argument("--output",
                        default=None,
                        help="where to write the graphed stream's voice")
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> None:
    args = parse_args(argv)
    isolator = voice_isolation.VoiceIsolator(args.model,
                                             device=args.device,
                                             mode="streaming",
                                             lookahead_frames=2)
    wav, sample_rate = voice_isolation.load_audio(args.recording)
    wav = voice_isolation.resample(wav, sample_rate, voice_isolation.SR)
    clips = wav[None].repeat(args.streams, 1)
    audio_seconds = clips.numel() / voice_isolation.SR
    plain, plain_seconds = stream(isolator, clips, cuda_graphs=False)
    graphed, graphed_seconds = stream(isolator, clips, cuda_graphs=True)
    report("plain", plain_seconds, audio_seconds)
    report("cuda_graphs", graphed_seconds, audio_seconds)
    print(f"largest difference between the two: "
          f"{np.abs(plain - graphed).max():.2e}")
    if args.output:
        voice_isolation.save_audio(args.output, graphed[0], voice_isolation.SR)


if __name__ == "__main__":
    main()
