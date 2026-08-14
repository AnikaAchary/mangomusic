"""Public MangoMusic exceptions."""


class MangoMusicError(Exception):
    """Base class for expected MangoMusic failures."""


class AudioDecodeError(MangoMusicError):
    """Raised when an audio input cannot be decoded into a usable signal."""
