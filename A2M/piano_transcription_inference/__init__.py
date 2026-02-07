from .config import sample_rate

__all__ = ["PianoTranscription", "sample_rate", "load_audio"]


def __getattr__(name):
    if name == "PianoTranscription":
        from .inference import PianoTranscription
        return PianoTranscription
    if name == "load_audio":
        from .utilities import load_audio
        return load_audio
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
