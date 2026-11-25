# Examples

## Audio cleanup notebook

[audio_cleanup.ipynb](audio_cleanup.ipynb) is a guided first run of the offline model, for evaluating it on your own audio:

1. Install the package and check the machine.
2. Load the offline model; its weights (256 MB) download from Hugging Face on first use.
3. Clean a bundled noisy sample: speech over a hum and sharp taps.
4. Play the original, the cleaned voice and the removed background side by side.
5. Swap in your own recording and save the output.

Run it from a clone:

```bash
git clone https://github.com/inworld-ai/voice-isolator && cd voice-isolator
pip install jupyterlab
jupyter lab examples/audio_cleanup.ipynb
```

It also runs in Colab or any other Jupyter environment without a clone: it then installs the package from PyPI and downloads the sample.

A GPU is optional. See [Hardware](../README.md#hardware) for what the model is tested on.

## The smallest reusable code

To clean a file from your own code:

```python
from inworld import voice_isolation

isolator = voice_isolation.VoiceIsolator("inworld/voice-isolator")  # load once, reuse
wav, sample_rate = voice_isolation.load_audio("noisy.wav")
voice = isolator.isolate(wav, sample_rate)
voice_isolation.save_audio("clean.wav", voice, sample_rate)
```

From a terminal, `voice-isolate noisy.wav clean.wav` does the same and also writes the removed background.

For live audio, see [Streaming](../README.md#streaming) and the [interactive demo](../demo/README.md).

## Streaming on an NVIDIA GPU

[stream_on_cuda.py](stream_on_cuda.py) plays a recording into a live session in 20 ms pieces, once as a plain stream and once with `cuda_graphs=True`, and prints how long each push took and how far the two outputs differ (they are equal). `--streams 8` streams eight copies as one batch, as a server with several live calls would:

```bash
python examples/stream_on_cuda.py demo/examples/sample_14.wav --streams 8
```

On CUDA, the scan and a stream's time attention run as Triton kernels (`inworld.voice_isolation.cuda`) whenever Triton is installed, as it is with PyTorch's CUDA builds on Linux; `cuda_graphs=True` also replays each block of the stream as one CUDA graph.
