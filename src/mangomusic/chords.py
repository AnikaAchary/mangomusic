"""Chord recognition from beat-synchronous chroma features."""

from enum import StrEnum
from itertools import pairwise
from typing import Self

import numpy as np
import numpy.typing as npt
from pydantic import BaseModel, Field, field_validator, model_validator

from mangomusic.chroma import PITCH_CLASS_COUNT, PITCH_CLASS_NAMES, normalize_chroma

CHORD_COUNT = 24
NO_CHORD_SYMBOL = "N"
DEFAULT_MIN_CONFIDENCE = 0.5

_MAJOR_INTERVALS = (0, 4, 7)
_MINOR_INTERVALS = (0, 3, 7)

# A score must clear the threshold by more than float32 rounding error before it
# counts as a chord. Without this, a beat scoring exactly the threshold — such as
# the flat chroma vector that scores the 0.5 default — would be labeled a chord or
# no-chord depending on which side of the threshold the hardware happened to round
# to. The margin is far below any musically meaningful difference in similarity.
_CONFIDENCE_TOLERANCE = 1e-6


class ChordQuality(StrEnum):
    """Triad quality of a recognized chord."""

    MAJOR = "maj"
    MINOR = "min"


class ChordLabel(BaseModel):
    """The chord recognized on a single beat."""

    beat_index: int = Field(ge=0)
    timestamp_seconds: float = Field(ge=0.0)
    root: str | None = None
    quality: ChordQuality | None = None
    confidence: float = Field(ge=0.0, le=1.0)

    @field_validator("root")
    @classmethod
    def _validate_root(cls, root: str | None) -> str | None:
        if root is not None and root not in PITCH_CLASS_NAMES:
            raise ValueError("root must name a pitch class")
        return root

    @model_validator(mode="after")
    def _validate_consistency(self) -> Self:
        if (self.root is None) != (self.quality is None):
            raise ValueError("root and quality must both be set or both be absent")
        return self

    @property
    def symbol(self) -> str:
        """Render the label as ``"<root>:<quality>"``, or ``"N"`` for no chord."""
        return _chord_symbol(self.root, self.quality)


def _empty_chord_labels() -> list[ChordLabel]:
    return []


class ChordAnalysis(BaseModel):
    """Serializable per-beat chord recognition output."""

    labels: list[ChordLabel] = Field(default_factory=_empty_chord_labels)
    min_confidence: float = Field(ge=0.0, le=1.0)

    @model_validator(mode="after")
    def _validate_labels(self) -> Self:
        for index, label in enumerate(self.labels):
            if label.beat_index != index:
                raise ValueError("labels must carry consecutive beat indices from zero")
        timestamps = [label.timestamp_seconds for label in self.labels]
        if any(later <= earlier for earlier, later in pairwise(timestamps)):
            raise ValueError("labels must have strictly increasing timestamps")
        return self


def chord_templates() -> npt.NDArray[np.float32]:
    """Build one binary pitch-class template per chord in the vocabulary.

    Each template carries equal weight on its root, third, and fifth and zero
    elsewhere, then is scaled to unit Euclidean length. Because chroma vectors
    are L2-normalized too, a dot product against a template is that chord's
    cosine similarity. Every template is a rotation of the C template of the
    same quality, so scoring is exactly transposition-equivariant.

    Returns:
        Unitless templates with shape ``(24, 12)`` and unit-norm rows. Row
        ``index`` names the chord at ``chord_label_names()[index]``: rows ``0``
        through ``11`` are the major triads rooted at C through B, and rows
        ``12`` through ``23`` the minor triads in the same root order. Columns
        are pitch classes ordered as in ``PITCH_CLASS_NAMES``. A fresh array is
        returned on each call, so callers may modify it freely.
    """
    templates = np.zeros((CHORD_COUNT, PITCH_CLASS_COUNT), dtype=np.float64)
    for quality_index, quality in enumerate(ChordQuality):
        intervals = _quality_intervals(quality)
        for root_index in range(PITCH_CLASS_COUNT):
            row = quality_index * PITCH_CLASS_COUNT + root_index
            for interval in intervals:
                templates[row, (root_index + interval) % PITCH_CLASS_COUNT] = 1.0
    norms = np.sqrt(np.sum(templates * templates, axis=1, keepdims=True))
    return np.asarray(templates / norms, dtype=np.float32)


def chord_label_names() -> tuple[str, ...]:
    """Name each chord template in row order.

    Returns:
        The 24 chord symbols, such as ``"C:maj"`` and ``"A:min"``, in the row
        order described in :func:`chord_templates`. The no-chord symbol is not
        included: it has no template and is assigned by confidence instead.
    """
    return tuple(
        _chord_symbol(root, quality)
        for quality in ChordQuality
        for root in PITCH_CLASS_NAMES
    )


def score_chord_templates(
    chroma: npt.NDArray[np.float32],
) -> npt.NDArray[np.float32]:
    """Score every chord template against every chroma vector.

    The input is L2-normalized first, so scores do not depend on how loud a
    frame is and un-normalized input is scored correctly.

    Args:
        chroma: Unitless, finite pitch-class energies with shape
            ``(12, frame_count)`` and at least one frame, ordered as described
            in :func:`mangomusic.chroma.compute_chroma`.

    Returns:
        Unitless cosine similarities in ``[0.0, 1.0]`` with shape
        ``(24, frame_count)``, where row ``index`` corresponds to
        ``chord_label_names()[index]``. A frame with no energy scores ``0.0``
        against every chord.

    Raises:
        ValueError: If ``chroma`` does not have a usable shape or contains
            non-finite values.
    """
    normalized = normalize_chroma(chroma).astype(np.float64, copy=False)
    templates = chord_templates().astype(np.float64, copy=False)
    scores = np.clip(templates @ normalized, 0.0, 1.0)
    return np.asarray(scores, dtype=np.float32)


def recognize_chords(
    beat_chroma: npt.NDArray[np.float32],
    beat_times_seconds: npt.NDArray[np.float64],
    *,
    min_confidence: float = DEFAULT_MIN_CONFIDENCE,
) -> ChordAnalysis:
    """Label each beat with its best-scoring chord.

    Every beat is labeled independently from its own chroma vector; no
    smoothing or continuity across beats is applied. A beat whose best score
    does not exceed ``min_confidence`` is labeled no-chord; a score sitting on
    the threshold within floating-point rounding error counts as no-chord, so
    the boundary does not shift between platforms. Ties are broken toward the
    lower template row, so the result is deterministic.

    Args:
        beat_chroma: Unitless, finite beat-synchronous pitch-class energies with
            shape ``(12, beat_count)``, as returned by
            :func:`mangomusic.chroma.aggregate_chroma_by_beat`.
        beat_times_seconds: Strictly increasing, nonnegative beat timestamps in
            seconds with shape ``(beat_count,)``, one per chroma column.
        min_confidence: Cosine similarity a beat must exceed to earn a chord,
            in ``[0.0, 1.0]``. The default of ``0.5`` is the score a flat chroma
            vector — one carrying no harmonic information — earns against every
            triad template, so a beat must beat a uniform spectrum to be
            labeled.

    Returns:
        One :class:`ChordLabel` per beat, in beat order, alongside the
        ``min_confidence`` that produced them. A label's confidence is its best
        template score whether or not that score cleared the threshold. An empty
        beat grid produces an analysis with no labels.

    Raises:
        ValueError: If the chroma matrix, beat timestamps, or minimum
            confidence are unusable.
    """
    if not 0.0 <= min_confidence <= 1.0:
        raise ValueError("min_confidence must lie in [0.0, 1.0]")

    if beat_chroma.shape == (PITCH_CLASS_COUNT, 0):
        _validate_beat_times(beat_times_seconds, 0)
        return ChordAnalysis(labels=[], min_confidence=min_confidence)

    scores = score_chord_templates(beat_chroma)
    beat_count = scores.shape[1]
    _validate_beat_times(beat_times_seconds, beat_count)

    best_rows = np.argmax(scores, axis=0)
    labels: list[ChordLabel] = []
    for beat_index in range(beat_count):
        row = int(best_rows[beat_index])
        confidence = float(scores[row, beat_index])
        root, quality = (
            _row_to_chord(row)
            if confidence > min_confidence + _CONFIDENCE_TOLERANCE
            else (None, None)
        )
        labels.append(
            ChordLabel(
                beat_index=beat_index,
                timestamp_seconds=float(beat_times_seconds[beat_index]),
                root=root,
                quality=quality,
                confidence=confidence,
            )
        )
    return ChordAnalysis(labels=labels, min_confidence=min_confidence)


def _quality_intervals(quality: ChordQuality) -> tuple[int, ...]:
    match quality:
        case ChordQuality.MAJOR:
            return _MAJOR_INTERVALS
        case ChordQuality.MINOR:
            return _MINOR_INTERVALS


def _chord_symbol(root: str | None, quality: ChordQuality | None) -> str:
    if root is None or quality is None:
        return NO_CHORD_SYMBOL
    return f"{root}:{quality.value}"


def _row_to_chord(row: int) -> tuple[str, ChordQuality]:
    qualities = tuple(ChordQuality)
    return (
        PITCH_CLASS_NAMES[row % PITCH_CLASS_COUNT],
        qualities[row // PITCH_CLASS_COUNT],
    )


def _validate_beat_times(
    beat_times_seconds: npt.NDArray[np.float64],
    beat_count: int,
) -> None:
    if beat_times_seconds.ndim != 1:
        raise ValueError("beat_times_seconds must be one-dimensional")
    if beat_times_seconds.size != beat_count:
        raise ValueError("beat_times_seconds must have one timestamp per beat")
    if not bool(np.all(np.isfinite(beat_times_seconds))):
        raise ValueError("beat_times_seconds must contain only finite values")
    if bool(np.any(beat_times_seconds < 0.0)):
        raise ValueError("beat_times_seconds must be nonnegative")
    if not bool(np.all(np.diff(beat_times_seconds) > 0.0)):
        raise ValueError("beat_times_seconds must be strictly increasing")
