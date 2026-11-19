"""Audio utilities: PCM conversion, resampling, microphone capture, playback."""
from afrilingua.audio.pcm import decode_audio, encode_wav, float32_to_pcm16, load_audio, pcm16_to_float32
from afrilingua.audio.resample import StreamResampler, resample

__all__ = [
    "StreamResampler",
    "decode_audio",
    "encode_wav",
    "float32_to_pcm16",
    "load_audio",
    "pcm16_to_float32",
    "resample",
]
