"""Chroma feature extraction and beat-synchronous aggregation."""

from enum import StrEnum

import librosa
import numpy as np
import numpy.typing as npt

PITCH_CLASS_NAMES: tuple[str, ...] = (
    "C",
    "C#",
    "D",
    "D#",
    "E",
    "F",
    "F#",
    "G",
    "G#",
    "A",
    "A#",
    "B",
)
PITCH_CLASS_COUNT = 12
DEFAULT_HOP_LENGTH_SAMPLES = 512

_BINS_PER_OCTAVE = 36
_A440_TUNING_SEMITONES = 0.0


class ChromaAggregation(StrEnum):
    """Strategy for collapsing the feature frames of one beat span."""

    MEDIAN = "median"
    MEAN = "mean"


def compute_chroma(
    samples: npt.NDArray[np.float32],
    sample_rate_hz: int,
    *,
    hop_length_samples: int = DEFAULT_HOP_LENGTH_SAMPLES,
) -> tuple[npt.NDArray[np.float32], npt.NDArray[np.float64]]:
    """Fold mono audio into per-frame pitch-class energy.

    Energy is summed across octaves by a constant-Q transform, so row ``0`` is
    pitch class C, row ``1`` is C#, and so on through row ``11`` for B, matching
    ``PITCH_CLASS_NAMES``. Pitch is referenced to A440 equal temperament; the
    tuning of the input is not estimated.

    Args:
        samples: Mono, finite audio amplitudes with shape ``(sample_count,)``.
            Amplitudes are unitless and expected in ``[-1.0, 1.0]``.
        sample_rate_hz: Audio sample rate in hertz. Must be positive.
        hop_length_samples: Frame advance in samples. Must be positive.

    Returns:
        Unitless pitch-class energies with shape ``(12, frame_count)``, each
        column L2-normalized as described in :func:`normalize_chroma`, and the
        frame timestamps in seconds with shape ``(frame_count,)``. Frame
        ``index`` is centered at ``index * hop_length_samples / sample_rate_hz``
        seconds.

    Raises:
        ValueError: If the sample rate, hop length, or audio samples are
            unusable.
    """
    _validate_audio(samples, sample_rate_hz)
    if hop_length_samples <= 0:
        raise ValueError("hop_length_samples must be positive")

    chroma = np.asarray(
        librosa.feature.chroma_cqt(
            y=samples,
            sr=sample_rate_hz,
            hop_length=hop_length_samples,
            bins_per_octave=_BINS_PER_OCTAVE,
            norm=None,
            tuning=_A440_TUNING_SEMITONES,
        ),
        dtype=np.float32,
    )
    frame_indices = np.arange(chroma.shape[1], dtype=np.float64)
    frame_times_seconds = (
        frame_indices * float(hop_length_samples) / float(sample_rate_hz)
    )
    return normalize_chroma(chroma), frame_times_seconds


def normalize_chroma(chroma: npt.NDArray[np.float32]) -> npt.NDArray[np.float32]:
    """Scale every chroma vector to unit Euclidean length.

    A dot product between two L2-normalized vectors is their cosine similarity,
    so normalizing both features and chord templates this way makes template
    scores directly comparable across frames of differing loudness. Columns that
    are entirely zero, such as those from silence, are left as zeros rather than
    being given an arbitrary direction.

    Args:
        chroma: Unitless, finite pitch-class energies with shape
            ``(12, frame_count)`` and at least one frame.

    Returns:
        Pitch-class energies with shape ``(12, frame_count)`` whose columns each
        have an L2 norm of ``1.0``, except all-zero columns, which stay zero.

    Raises:
        ValueError: If ``chroma`` does not have a usable shape or contains
            non-finite values.
    """
    _validate_chroma(chroma)
    values = chroma.astype(np.float64, copy=True)
    norms = np.sqrt(np.sum(values * values, axis=0))
    scales = np.where(norms > 0.0, norms, 1.0)
    return np.asarray(values / scales, dtype=np.float32)


def aggregate_chroma_by_beat(
    chroma: npt.NDArray[np.float32],
    frame_times_seconds: npt.NDArray[np.float64],
    beat_times_seconds: npt.NDArray[np.float64],
    *,
    aggregation: ChromaAggregation = ChromaAggregation.MEDIAN,
) -> npt.NDArray[np.float32]:
    """Collapse the feature frames of each beat into one chroma vector.

    Beat ``index`` spans ``[beat_times_seconds[index],
    beat_times_seconds[index + 1])``, and the final beat spans from its own
    timestamp through the last feature frame, so the result always holds one
    vector per beat. A span containing no feature frame — possible when beats
    are closer together than the hop length — falls back to the single frame
    nearest the span midpoint, so every beat carries real harmonic content.

    Args:
        chroma: Unitless, finite pitch-class energies with shape
            ``(12, frame_count)`` and at least one frame, ordered as described
            in :func:`compute_chroma`.
        frame_times_seconds: Strictly increasing frame timestamps in seconds
            with shape ``(frame_count,)``.
        beat_times_seconds: Strictly increasing, nonnegative beat timestamps in
            seconds with shape ``(beat_count,)``. Must not be empty.
        aggregation: Reduction applied to the frames of each span. The default
            median ignores a minority of outlying frames, such as the broadband
            energy of a note attack, that would pull the mean.

    Returns:
        Unitless beat-synchronous pitch-class energies with shape
        ``(12, beat_count)``, each column L2-normalized as described in
        :func:`normalize_chroma`. A column is all zeros when its span carries no
        energy, and, under the median, when no pitch class sounds in a majority
        of the span's frames.

    Raises:
        ValueError: If the chroma matrix, frame timestamps, or beat timestamps
            are unusable.
    """
    _validate_chroma(chroma)
    _validate_frame_times(chroma, frame_times_seconds)
    _validate_beat_times(beat_times_seconds)

    span_starts_seconds = beat_times_seconds.astype(np.float64, copy=False)
    span_stops_seconds = np.concatenate(
        [span_starts_seconds[1:], np.array([np.inf], dtype=np.float64)]
    )
    start_indices = np.searchsorted(
        frame_times_seconds, span_starts_seconds, side="left"
    )
    stop_indices = np.searchsorted(frame_times_seconds, span_stops_seconds, side="left")

    values = chroma.astype(np.float64, copy=False)
    aggregated = np.empty(
        (PITCH_CLASS_COUNT, span_starts_seconds.size),
        dtype=np.float64,
    )
    for index in range(span_starts_seconds.size):
        start = int(start_indices[index])
        stop = int(stop_indices[index])
        if stop > start:
            aggregated[:, index] = _reduce_span(values[:, start:stop], aggregation)
        else:
            nearest = _nearest_frame_index(
                frame_times_seconds,
                float(span_starts_seconds[index]),
                float(span_stops_seconds[index]),
            )
            aggregated[:, index] = values[:, nearest]

    return normalize_chroma(np.asarray(aggregated, dtype=np.float32))


def _reduce_span(
    span: npt.NDArray[np.float64],
    aggregation: ChromaAggregation,
) -> npt.NDArray[np.float64]:
    match aggregation:
        case ChromaAggregation.MEDIAN:
            return np.asarray(np.median(span, axis=1), dtype=np.float64)
        case ChromaAggregation.MEAN:
            return np.asarray(np.mean(span, axis=1), dtype=np.float64)


def _nearest_frame_index(
    frame_times_seconds: npt.NDArray[np.float64],
    span_start_seconds: float,
    span_stop_seconds: float,
) -> int:
    reference_seconds = (
        0.5 * (span_start_seconds + span_stop_seconds)
        if np.isfinite(span_stop_seconds)
        else span_start_seconds
    )
    return int(np.argmin(np.abs(frame_times_seconds - reference_seconds)))


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


def _validate_chroma(chroma: npt.NDArray[np.float32]) -> None:
    if chroma.ndim != 2 or chroma.shape[0] != PITCH_CLASS_COUNT:
        raise ValueError("chroma must have shape (12, frame_count)")
    if chroma.shape[1] == 0:
        raise ValueError("chroma must contain at least one frame")
    if not bool(np.all(np.isfinite(chroma))):
        raise ValueError("chroma must contain only finite values")


def _validate_frame_times(
    chroma: npt.NDArray[np.float32],
    frame_times_seconds: npt.NDArray[np.float64],
) -> None:
    if frame_times_seconds.ndim != 1:
        raise ValueError("frame_times_seconds must be one-dimensional")
    if frame_times_seconds.size != chroma.shape[1]:
        raise ValueError("frame_times_seconds must have one timestamp per chroma frame")
    if not bool(np.all(np.isfinite(frame_times_seconds))):
        raise ValueError("frame_times_seconds must contain only finite values")
    if not bool(np.all(np.diff(frame_times_seconds) > 0.0)):
        raise ValueError("frame_times_seconds must be strictly increasing")


def _validate_beat_times(beat_times_seconds: npt.NDArray[np.float64]) -> None:
    if beat_times_seconds.ndim != 1:
        raise ValueError("beat_times_seconds must be one-dimensional")
    if beat_times_seconds.size == 0:
        raise ValueError("beat_times_seconds must not be empty")
    if not bool(np.all(np.isfinite(beat_times_seconds))):
        raise ValueError("beat_times_seconds must contain only finite values")
    if bool(np.any(beat_times_seconds < 0.0)):
        raise ValueError("beat_times_seconds must be nonnegative")
    if not bool(np.all(np.diff(beat_times_seconds) > 0.0)):
        raise ValueError("beat_times_seconds must be strictly increasing")
