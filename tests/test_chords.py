"""Tests for template-based chord recognition."""

from collections.abc import Sequence

import numpy as np
import numpy.typing as npt
import pytest
from pydantic import ValidationError

from mangomusic.chords import (
    CHORD_COUNT,
    DEFAULT_MIN_CONFIDENCE,
    NO_CHORD_SYMBOL,
    ChordAnalysis,
    ChordLabel,
    ChordQuality,
    chord_label_names,
    chord_templates,
    recognize_chords,
    score_chord_templates,
)
from mangomusic.chroma import (
    PITCH_CLASS_COUNT,
    PITCH_CLASS_NAMES,
    aggregate_chroma_by_beat,
    compute_chroma,
)

_SAMPLE_RATE_HZ = 22_050

_C_MAJOR_HZ = (261.63, 329.63, 392.00)
_A_MINOR_HZ = (440.00, 523.25, 659.26)

_MAJOR_INTERVALS = (0, 4, 7)
_MINOR_INTERVALS = (0, 3, 7)


def _triad(
    frequencies_hz: Sequence[float],
    duration_seconds: float,
    sample_rate_hz: int = _SAMPLE_RATE_HZ,
) -> npt.NDArray[np.float32]:
    """Sum equal-amplitude sine partials into a mono triad."""
    times_seconds = np.arange(round(sample_rate_hz * duration_seconds)) / sample_rate_hz
    tones = [
        np.sin(2.0 * np.pi * frequency_hz * times_seconds)
        for frequency_hz in frequencies_hz
    ]
    return np.asarray(
        0.3 * np.sum(tones, axis=0) / len(frequencies_hz),
        dtype=np.float32,
    )


def _chroma_column(pitch_classes: Sequence[int]) -> npt.NDArray[np.float32]:
    """Build a single unit-weight chroma column over the given pitch classes."""
    column = np.zeros((PITCH_CLASS_COUNT, 1), dtype=np.float32)
    column[list(pitch_classes), 0] = 1.0
    return column


def _triad_pitch_classes(root_index: int, quality: ChordQuality) -> tuple[int, ...]:
    intervals = _MAJOR_INTERVALS if quality is ChordQuality.MAJOR else _MINOR_INTERVALS
    return tuple((root_index + interval) % PITCH_CLASS_COUNT for interval in intervals)


def _recognize_single(
    chroma: npt.NDArray[np.float32],
    *,
    min_confidence: float = DEFAULT_MIN_CONFIDENCE,
) -> ChordLabel:
    analysis = recognize_chords(
        chroma,
        np.array([0.0], dtype=np.float64),
        min_confidence=min_confidence,
    )
    return analysis.labels[0]


def test_chord_templates_have_expected_shape_and_unit_norms() -> None:
    templates = chord_templates()

    assert templates.shape == (CHORD_COUNT, PITCH_CLASS_COUNT)
    np.testing.assert_allclose(
        np.linalg.norm(templates.astype(np.float64), axis=1),
        np.ones(CHORD_COUNT),
        atol=1e-6,
    )


@pytest.mark.parametrize("quality", list(ChordQuality))
@pytest.mark.parametrize("root_index", range(PITCH_CLASS_COUNT))
def test_chord_templates_hold_their_triad_pitch_classes(
    root_index: int,
    quality: ChordQuality,
) -> None:
    templates = chord_templates()
    row = list(ChordQuality).index(quality) * PITCH_CLASS_COUNT + root_index

    sounding = set(np.flatnonzero(templates[row]).tolist())

    assert sounding == set(_triad_pitch_classes(root_index, quality))


def test_chord_label_names_match_template_rows() -> None:
    names = chord_label_names()

    assert len(names) == CHORD_COUNT
    assert names[0] == "C:maj"
    assert names[PITCH_CLASS_COUNT - 1] == "B:maj"
    assert names[PITCH_CLASS_COUNT] == "C:min"
    assert names[CHORD_COUNT - 1] == "B:min"
    assert NO_CHORD_SYMBOL not in names


def test_templates_are_rotations_of_the_c_template() -> None:
    templates = chord_templates()

    for quality_index in range(len(ChordQuality)):
        base_row = quality_index * PITCH_CLASS_COUNT
        for root_index in range(PITCH_CLASS_COUNT):
            np.testing.assert_allclose(
                templates[base_row + root_index],
                np.roll(templates[base_row], root_index),
                atol=1e-6,
            )


def test_recognizes_a_synthesized_major_triad() -> None:
    samples = _triad(_C_MAJOR_HZ, 1.0)
    chroma, frame_times_seconds = compute_chroma(samples, _SAMPLE_RATE_HZ)
    beat_times_seconds = np.array([0.0, 0.5], dtype=np.float64)
    beat_chroma = aggregate_chroma_by_beat(
        chroma,
        frame_times_seconds,
        beat_times_seconds,
    )

    analysis = recognize_chords(beat_chroma, beat_times_seconds)

    assert [label.symbol for label in analysis.labels] == ["C:maj", "C:maj"]


def test_recognizes_a_synthesized_minor_triad() -> None:
    samples = _triad(_A_MINOR_HZ, 1.0)
    chroma, frame_times_seconds = compute_chroma(samples, _SAMPLE_RATE_HZ)
    beat_times_seconds = np.array([0.0, 0.5], dtype=np.float64)
    beat_chroma = aggregate_chroma_by_beat(
        chroma,
        frame_times_seconds,
        beat_times_seconds,
    )

    analysis = recognize_chords(beat_chroma, beat_times_seconds)

    assert [label.symbol for label in analysis.labels] == ["A:min", "A:min"]


def test_major_triad_is_not_confused_with_its_relative_or_mediant_minor() -> None:
    """C major shares two of three tones with A minor and E minor."""
    label = _recognize_single(
        _chroma_column(_triad_pitch_classes(0, ChordQuality.MAJOR))
    )

    assert label.symbol == "C:maj"
    assert label.symbol not in {"A:min", "E:min"}


@pytest.mark.parametrize("quality", list(ChordQuality))
@pytest.mark.parametrize("root_index", range(PITCH_CLASS_COUNT))
def test_recognition_is_transposition_equivariant(
    root_index: int,
    quality: ChordQuality,
) -> None:
    label = _recognize_single(_chroma_column(_triad_pitch_classes(root_index, quality)))

    assert label.root == PITCH_CLASS_NAMES[root_index]
    assert label.quality is quality
    np.testing.assert_allclose(label.confidence, 1.0, atol=1e-6)


def test_silent_beat_is_labeled_no_chord() -> None:
    silent = np.zeros((PITCH_CLASS_COUNT, 1), dtype=np.float32)

    label = _recognize_single(silent)

    assert label.symbol == NO_CHORD_SYMBOL
    assert label.root is None
    assert label.quality is None
    np.testing.assert_allclose(label.confidence, 0.0, atol=1e-6)


def test_flat_chroma_is_labeled_no_chord_at_the_default_threshold() -> None:
    """A flat vector carries no harmonic information and scores 0.5 everywhere."""
    flat = np.full((PITCH_CLASS_COUNT, 1), 1.0, dtype=np.float32)

    scores = score_chord_templates(flat)
    np.testing.assert_allclose(
        scores[:, 0].astype(np.float64),
        np.full(CHORD_COUNT, DEFAULT_MIN_CONFIDENCE),
        atol=1e-6,
    )
    assert _recognize_single(flat).symbol == NO_CHORD_SYMBOL


def test_lowering_the_threshold_labels_an_otherwise_rejected_beat() -> None:
    flat = np.full((PITCH_CLASS_COUNT, 1), 1.0, dtype=np.float32)

    label = _recognize_single(flat, min_confidence=0.25)

    assert label.symbol != NO_CHORD_SYMBOL


@pytest.mark.parametrize("offset", [-1e-9, 0.0, 1e-9])
def test_a_score_sitting_on_the_threshold_is_no_chord(offset: float) -> None:
    """The 0.5 boundary must not move with floating-point rounding."""
    flat = np.full((PITCH_CLASS_COUNT, 1), 1.0, dtype=np.float32)

    label = _recognize_single(flat, min_confidence=DEFAULT_MIN_CONFIDENCE + offset)

    assert label.symbol == NO_CHORD_SYMBOL


def test_confidence_is_reported_even_when_below_the_threshold() -> None:
    flat = np.full((PITCH_CLASS_COUNT, 1), 1.0, dtype=np.float32)

    label = _recognize_single(flat)

    assert label.symbol == NO_CHORD_SYMBOL
    np.testing.assert_allclose(label.confidence, DEFAULT_MIN_CONFIDENCE, atol=1e-6)


def test_ties_resolve_to_the_lowest_template_row() -> None:
    """An augmented triad shares two tones with six templates; C:maj is lowest."""
    augmented = _chroma_column((0, 4, 8))

    label = _recognize_single(augmented)

    assert label.symbol == "C:maj"
    np.testing.assert_allclose(label.confidence, 2.0 / 3.0, atol=1e-6)


def test_returns_one_label_per_beat_with_input_timestamps() -> None:
    chroma = np.concatenate(
        [
            _chroma_column(_triad_pitch_classes(0, ChordQuality.MAJOR)),
            _chroma_column(_triad_pitch_classes(9, ChordQuality.MINOR)),
            _chroma_column(_triad_pitch_classes(5, ChordQuality.MAJOR)),
        ],
        axis=1,
    )
    beat_times_seconds = np.array([0.0, 0.5, 1.25], dtype=np.float64)

    analysis = recognize_chords(chroma, beat_times_seconds)

    assert [label.symbol for label in analysis.labels] == ["C:maj", "A:min", "F:maj"]
    assert [label.beat_index for label in analysis.labels] == [0, 1, 2]
    np.testing.assert_allclose(
        [label.timestamp_seconds for label in analysis.labels],
        beat_times_seconds,
        atol=1e-9,
    )
    np.testing.assert_allclose(
        analysis.min_confidence,
        DEFAULT_MIN_CONFIDENCE,
        atol=1e-12,
    )


def test_scoring_ignores_chroma_scale() -> None:
    chroma = _chroma_column(_triad_pitch_classes(7, ChordQuality.MAJOR))

    np.testing.assert_allclose(
        score_chord_templates(chroma).astype(np.float64),
        score_chord_templates(np.asarray(chroma * 17.0, dtype=np.float32)).astype(
            np.float64
        ),
        atol=1e-6,
    )


def test_recognition_is_deterministic() -> None:
    samples = _triad(_C_MAJOR_HZ, 1.0)
    chroma, frame_times_seconds = compute_chroma(samples, _SAMPLE_RATE_HZ)
    beat_times_seconds = np.array([0.0, 0.5], dtype=np.float64)
    beat_chroma = aggregate_chroma_by_beat(
        chroma,
        frame_times_seconds,
        beat_times_seconds,
    )

    first = recognize_chords(beat_chroma, beat_times_seconds)
    second = recognize_chords(beat_chroma, beat_times_seconds)

    assert first.model_dump_json() == second.model_dump_json()


def test_analysis_round_trips_through_json() -> None:
    chroma = _chroma_column(_triad_pitch_classes(2, ChordQuality.MINOR))
    analysis = recognize_chords(chroma, np.array([0.75], dtype=np.float64))

    restored = ChordAnalysis.model_validate_json(analysis.model_dump_json())

    assert restored == analysis
    assert restored.labels[0].symbol == "D:min"


@pytest.mark.parametrize("min_confidence", [-0.1, 1.1])
def test_recognize_chords_rejects_out_of_range_threshold(
    min_confidence: float,
) -> None:
    chroma = _chroma_column((0, 4, 7))

    with pytest.raises(ValueError, match="min_confidence"):
        recognize_chords(
            chroma,
            np.array([0.0], dtype=np.float64),
            min_confidence=min_confidence,
        )


@pytest.mark.parametrize(
    ("chroma", "message"),
    [
        (np.zeros((11, 4), dtype=np.float32), "shape"),
        (np.zeros((PITCH_CLASS_COUNT, 0), dtype=np.float32), "at least one frame"),
        (
            np.full((PITCH_CLASS_COUNT, 2), np.nan, dtype=np.float32),
            "finite",
        ),
    ],
)
def test_score_chord_templates_rejects_unusable_chroma(
    chroma: npt.NDArray[np.float32],
    message: str,
) -> None:
    with pytest.raises(ValueError, match=message):
        score_chord_templates(chroma)


@pytest.mark.parametrize(
    ("beat_times_seconds", "message"),
    [
        (np.array([[0.0, 1.0]], dtype=np.float64), "one-dimensional"),
        (np.array([0.0], dtype=np.float64), "one timestamp per beat"),
        (np.array([0.0, np.inf], dtype=np.float64), "finite"),
        (np.array([-1.0, 0.5], dtype=np.float64), "nonnegative"),
        (np.array([1.0, 0.5], dtype=np.float64), "strictly increasing"),
    ],
)
def test_recognize_chords_rejects_unusable_beat_times(
    beat_times_seconds: npt.NDArray[np.float64],
    message: str,
) -> None:
    chroma = np.concatenate(
        [_chroma_column((0, 4, 7)), _chroma_column((5, 9, 0))],
        axis=1,
    )

    with pytest.raises(ValueError, match=message):
        recognize_chords(chroma, beat_times_seconds)


@pytest.mark.parametrize("confidence", [-0.1, 1.1])
def test_chord_label_rejects_out_of_range_confidence(confidence: float) -> None:
    with pytest.raises(ValidationError):
        ChordLabel(
            beat_index=0,
            timestamp_seconds=0.0,
            root="C",
            quality=ChordQuality.MAJOR,
            confidence=confidence,
        )


def test_chord_label_rejects_negative_beat_index() -> None:
    with pytest.raises(ValidationError):
        ChordLabel(
            beat_index=-1,
            timestamp_seconds=0.0,
            root="C",
            quality=ChordQuality.MAJOR,
            confidence=1.0,
        )


def test_chord_label_rejects_an_unknown_root() -> None:
    with pytest.raises(ValidationError):
        ChordLabel(
            beat_index=0,
            timestamp_seconds=0.0,
            root="H",
            quality=ChordQuality.MAJOR,
            confidence=1.0,
        )


@pytest.mark.parametrize(
    ("root", "quality"),
    [("C", None), (None, ChordQuality.MAJOR)],
)
def test_chord_label_rejects_a_half_set_chord(
    root: str | None,
    quality: ChordQuality | None,
) -> None:
    with pytest.raises(ValidationError):
        ChordLabel(
            beat_index=0,
            timestamp_seconds=0.0,
            root=root,
            quality=quality,
            confidence=1.0,
        )


def test_chord_analysis_rejects_non_consecutive_beat_indices() -> None:
    labels = [
        ChordLabel(
            beat_index=index,
            timestamp_seconds=float(index),
            root="C",
            quality=ChordQuality.MAJOR,
            confidence=1.0,
        )
        for index in (0, 2)
    ]

    with pytest.raises(ValidationError):
        ChordAnalysis(labels=labels, min_confidence=DEFAULT_MIN_CONFIDENCE)


def test_chord_analysis_rejects_non_increasing_timestamps() -> None:
    labels = [
        ChordLabel(
            beat_index=index,
            timestamp_seconds=timestamp_seconds,
            root="C",
            quality=ChordQuality.MAJOR,
            confidence=1.0,
        )
        for index, timestamp_seconds in enumerate((1.0, 0.5))
    ]

    with pytest.raises(ValidationError):
        ChordAnalysis(labels=labels, min_confidence=DEFAULT_MIN_CONFIDENCE)
