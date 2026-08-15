"""Rhythm feature extraction and beat tracking."""

from typing import Protocol, Self, cast

import librosa
import numpy as np
import numpy.typing as npt
from pydantic import BaseModel, Field, model_validator

_HOP_LENGTH_SAMPLES = 256
_SILENCE_RMS_THRESHOLD = 1e-5
_MIN_TRACKED_BEATS = 2


class _OnsetApi(Protocol):
    def onset_strength(
        self,
        *,
        y: npt.NDArray[np.float32],
        sr: int,
        hop_length: int,
    ) -> npt.NDArray[np.float32]: ...


_ONSET_API = cast(_OnsetApi, librosa.onset)


class BeatEvent(BaseModel):
    """A detected beat and its position in an assumed 4/4 bar."""

    timestamp_seconds: float = Field(ge=0.0)
    bar_number: int = Field(ge=1)
    beat_in_bar: int = Field(ge=1, le=4)
    confidence: float = Field(ge=0.0, le=1.0)


def _empty_beat_events() -> list[BeatEvent]:
    return []


class RhythmAnalysis(BaseModel):
    """Serializable global tempo and beat-tracking output."""

    bpm: float | None = Field(default=None, gt=0.0)
    tempo_confidence: float = Field(ge=0.0, le=1.0)
    beats: list[BeatEvent] = Field(default_factory=_empty_beat_events)
    is_silent: bool

    @model_validator(mode="after")
    def _validate_consistency(self) -> Self:
        if self.bpm is None:
            if self.beats or self.tempo_confidence != 0.0:
                raise ValueError(
                    "analysis without a BPM cannot contain beats or confidence"
                )
            return self

        if self.is_silent:
            raise ValueError("silent analysis cannot contain a BPM")
        if not self.beats:
            raise ValueError("analysis with a BPM must contain beats")
        return self


def compute_onset_strength(
    samples: npt.NDArray[np.float32],
    sample_rate_hz: int,
) -> tuple[npt.NDArray[np.float32], npt.NDArray[np.float64]]:
    """Compute a spectral-flux rhythm representation for mono audio.

    Args:
        samples: Mono, finite audio amplitudes with shape ``(sample_count,)``.
            Amplitudes are unitless and expected in ``[-1.0, 1.0]``.
        sample_rate_hz: Audio sample rate in hertz. Must be positive.

    Returns:
        Unitless onset-strength values and their timestamps in seconds. Both
        arrays have shape ``(onset_frame_count,)``.

    Raises:
        ValueError: If the sample rate or audio samples are unusable.
    """
    _validate_audio(samples, sample_rate_hz)
    return _compute_onset_strength(samples, sample_rate_hz)


def analyze_rhythm(
    samples: npt.NDArray[np.float32],
    sample_rate_hz: int,
) -> RhythmAnalysis:
    """Estimate global tempo and ordered beat times for mono audio.

    The first detected beat is treated as beat one of bar one, and subsequent
    beats are grouped into 4/4 bars. This does not infer the musical downbeat.

    Args:
        samples: Mono, finite audio amplitudes with shape ``(sample_count,)``.
            Amplitudes are unitless and expected in ``[-1.0, 1.0]``.
        sample_rate_hz: Audio sample rate in hertz. Must be positive.

    Returns:
        Global tempo in beats per minute, confidence values in ``[0.0, 1.0]``,
        and beat timestamps in seconds. Silent or non-rhythmic audio has no BPM
        or beats and has zero tempo confidence.

    Raises:
        ValueError: If the sample rate or audio samples are unusable.
    """
    _validate_audio(samples, sample_rate_hz)
    if _root_mean_square(samples) <= _SILENCE_RMS_THRESHOLD:
        return RhythmAnalysis(
            bpm=None,
            tempo_confidence=0.0,
            beats=[],
            is_silent=True,
        )

    onset_strength, onset_times_seconds = _compute_onset_strength(
        samples,
        sample_rate_hz,
    )
    tempo, beat_frames = librosa.beat.beat_track(
        onset_envelope=onset_strength,
        sr=sample_rate_hz,
        hop_length=_HOP_LENGTH_SAMPLES,
        sparse=True,
        units="frames",
    )
    bpm = _scalar_tempo(tempo)
    frames = np.asarray(beat_frames, dtype=np.int64).reshape(-1)
    frames = frames[(frames >= 0) & (frames < onset_strength.size)]
    frames = np.unique(frames)

    if bpm <= 0.0 or frames.size < _MIN_TRACKED_BEATS:
        return RhythmAnalysis(
            bpm=None,
            tempo_confidence=0.0,
            beats=[],
            is_silent=False,
        )

    beat_confidences = _beat_confidences(onset_strength, frames)
    beat_times_seconds = onset_times_seconds[frames]
    beats = [
        BeatEvent(
            timestamp_seconds=float(timestamp_seconds),
            bar_number=index // 4 + 1,
            beat_in_bar=index % 4 + 1,
            confidence=float(beat_confidences[index]),
        )
        for index, timestamp_seconds in enumerate(beat_times_seconds)
    ]
    return RhythmAnalysis(
        bpm=bpm,
        tempo_confidence=_tempo_confidence(
            bpm,
            beat_times_seconds,
            beat_confidences,
        ),
        beats=beats,
        is_silent=False,
    )


def _validate_audio(
    samples: npt.NDArray[np.float32],
    sample_rate_hz: int,
) -> None:
    if sample_rate_hz <= 0:
        raise ValueError("sample_rate_hz must be positive")
    if samples.ndim != 1:
        raise ValueError("samples must be a one-dimensional mono signal")
    if samples.size == 0:
        raise ValueError("samples must not be empty")
    if not bool(np.all(np.isfinite(samples))):
        raise ValueError("samples must contain only finite values")


def _compute_onset_strength(
    samples: npt.NDArray[np.float32],
    sample_rate_hz: int,
) -> tuple[npt.NDArray[np.float32], npt.NDArray[np.float64]]:
    strengths = np.asarray(
        _ONSET_API.onset_strength(
            y=samples,
            sr=sample_rate_hz,
            hop_length=_HOP_LENGTH_SAMPLES,
        ),
        dtype=np.float32,
    )
    frame_indices = np.arange(strengths.size, dtype=np.float64)
    timestamps_seconds = (
        frame_indices * float(_HOP_LENGTH_SAMPLES) / float(sample_rate_hz)
    )
    return strengths, timestamps_seconds


def _root_mean_square(samples: npt.NDArray[np.float32]) -> float:
    values = samples.astype(np.float64, copy=False)
    return float(np.sqrt(np.mean(values * values)))


def _scalar_tempo(tempo: float | npt.NDArray[np.float64]) -> float:
    values = np.asarray(tempo, dtype=np.float64).reshape(-1)
    if values.size == 0:
        return 0.0
    return float(values[0])


def _beat_confidences(
    onset_strength: npt.NDArray[np.float32],
    beat_frames: npt.NDArray[np.int64],
) -> npt.NDArray[np.float64]:
    positive_strengths = onset_strength[onset_strength > 0.0]
    if positive_strengths.size == 0:
        return np.zeros(beat_frames.size, dtype=np.float64)

    reference_strength = float(np.percentile(positive_strengths, 95.0))
    if reference_strength <= 0.0:
        return np.zeros(beat_frames.size, dtype=np.float64)

    confidences = np.empty(beat_frames.size, dtype=np.float64)
    for index, frame in enumerate(beat_frames):
        start = max(0, int(frame) - 2)
        stop = min(onset_strength.size, int(frame) + 3)
        local_strength = float(np.max(onset_strength[start:stop]))
        confidences[index] = np.clip(local_strength / reference_strength, 0.0, 1.0)
    return confidences


def _tempo_confidence(
    bpm: float,
    beat_times_seconds: npt.NDArray[np.float64],
    beat_confidences: npt.NDArray[np.float64],
) -> float:
    expected_interval_seconds = 60.0 / bpm
    intervals_seconds = np.diff(beat_times_seconds)
    relative_error = float(
        np.median(
            np.abs(intervals_seconds - expected_interval_seconds)
            / expected_interval_seconds
        )
    )
    timing_consistency = float(np.clip(1.0 - 4.0 * relative_error, 0.0, 1.0))
    onset_support = float(np.mean(beat_confidences))
    evidence = min(1.0, beat_times_seconds.size / 4.0)
    return float(np.clip(timing_consistency * onset_support * evidence, 0.0, 1.0))
