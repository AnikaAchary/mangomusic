"""Tests for chroma extraction and beat-synchronous aggregation."""

from collections.abc import Sequence

import numpy as np
import numpy.typing as npt
import pytest

from mangomusic.chroma import (
    DEFAULT_HOP_LENGTH_SAMPLES,
    PITCH_CLASS_COUNT,
    PITCH_CLASS_NAMES,
    ChromaAggregation,
    aggregate_chroma_by_beat,
    compute_chroma,
    normalize_chroma,
)

_SAMPLE_RATE_HZ = 22_050

_C_MAJOR_HZ = (261.63, 329.63, 392.00)
_F_MAJOR_HZ = (349.23, 440.00, 523.25)
_F_SHARP_MAJOR_HZ = (369.99, 466.16, 554.37)
_A_MINOR_HZ = (440.00, 523.25, 659.26)


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


def _pitch_class_vector(pitch_classes: Sequence[int]) -> npt.NDArray[np.float32]:
    """Build a unit-weight chroma column over the given pitch classes."""
    vector = np.zeros(PITCH_CLASS_COUNT, dtype=np.float32)
    vector[list(pitch_classes)] = 1.0
    return vector


def _column_norms(chroma: npt.NDArray[np.float32]) -> npt.NDArray[np.float64]:
    return np.sqrt(np.sum(chroma.astype(np.float64) ** 2, axis=0))


def _strongest_pitch_classes(
    chroma_vector: npt.NDArray[np.float64] | npt.NDArray[np.float32],
    count: int = 3,
) -> set[int]:
    return {int(index) for index in np.argsort(chroma_vector)[-count:]}


def _ramp_chroma(frame_count: int) -> npt.NDArray[np.float32]:
    """Deterministic chroma whose column ``index`` is one-hot on pitch class ``index``."""
    chroma = np.zeros((PITCH_CLASS_COUNT, frame_count), dtype=np.float32)
    for index in range(frame_count):
        chroma[index % PITCH_CLASS_COUNT, index] = 1.0
    return chroma


def _sustained_chroma(frame_count: int) -> npt.NDArray[np.float32]:
    """Chroma holding one triad while a passing tone moves frame to frame."""
    chroma = np.zeros((PITCH_CLASS_COUNT, frame_count), dtype=np.float32)
    chroma[[0, 4, 7], :] = 1.0
    for index in range(frame_count):
        chroma[(index * 5) % PITCH_CLASS_COUNT, index] += 0.2
    return chroma


def test_pitch_class_names_cover_the_octave() -> None:
    assert len(PITCH_CLASS_NAMES) == PITCH_CLASS_COUNT
    assert PITCH_CLASS_NAMES[0] == "C"
    assert PITCH_CLASS_NAMES[9] == "A"


@pytest.mark.parametrize(
    "frequencies_hz, expected_pitch_classes",
    [
        (_C_MAJOR_HZ, {0, 4, 7}),
        (_F_SHARP_MAJOR_HZ, {6, 10, 1}),
        (_A_MINOR_HZ, {9, 0, 4}),
    ],
)
def test_compute_chroma_emphasizes_triad_pitch_classes(
    frequencies_hz: tuple[float, ...],
    expected_pitch_classes: set[int],
) -> None:
    samples = _triad(frequencies_hz, duration_seconds=2.0)

    chroma, _ = compute_chroma(samples, _SAMPLE_RATE_HZ)
    summary = np.median(chroma.astype(np.float64), axis=1)

    assert _strongest_pitch_classes(summary) == expected_pitch_classes

    chord_tone_energies = summary[sorted(expected_pitch_classes)]
    other_energies = np.delete(summary, sorted(expected_pitch_classes))
    assert float(np.min(chord_tone_energies)) > 5.0 * float(np.max(other_energies))


@pytest.mark.parametrize("hop_length_samples", [256, DEFAULT_HOP_LENGTH_SAMPLES])
def test_compute_chroma_frame_times_follow_the_hop_grid(
    hop_length_samples: int,
) -> None:
    samples = _triad(_C_MAJOR_HZ, duration_seconds=1.0)

    chroma, frame_times_seconds = compute_chroma(
        samples,
        _SAMPLE_RATE_HZ,
        hop_length_samples=hop_length_samples,
    )

    frame_count = chroma.shape[1]
    assert chroma.shape == (PITCH_CLASS_COUNT, frame_count)
    assert frame_times_seconds.shape == (frame_count,)
    assert chroma.dtype == np.dtype(np.float32)
    assert frame_times_seconds.dtype == np.dtype(np.float64)
    assert frame_count > 0
    assert bool(np.all(np.diff(frame_times_seconds) > 0.0))

    expected_times_seconds = (
        np.arange(frame_count, dtype=np.float64) * hop_length_samples / _SAMPLE_RATE_HZ
    )
    np.testing.assert_allclose(
        frame_times_seconds,
        expected_times_seconds,
        rtol=0.0,
        atol=1e-12,
    )


def test_compute_chroma_normalizes_every_frame() -> None:
    samples = _triad(_C_MAJOR_HZ, duration_seconds=1.0)

    chroma, _ = compute_chroma(samples, _SAMPLE_RATE_HZ)

    np.testing.assert_allclose(
        _column_norms(chroma),
        np.ones(chroma.shape[1]),
        rtol=0.0,
        atol=1e-6,
    )


def test_compute_chroma_of_silence_is_all_zero() -> None:
    samples = np.zeros(_SAMPLE_RATE_HZ, dtype=np.float32)

    chroma, _ = compute_chroma(samples, _SAMPLE_RATE_HZ)

    assert not bool(np.any(np.isnan(chroma)))
    np.testing.assert_allclose(
        chroma,
        np.zeros_like(chroma),
        rtol=0.0,
        atol=0.0,
    )


def test_compute_chroma_is_deterministic() -> None:
    samples = _triad(_C_MAJOR_HZ, duration_seconds=1.0)

    first_chroma, first_times_seconds = compute_chroma(samples, _SAMPLE_RATE_HZ)
    second_chroma, second_times_seconds = compute_chroma(samples, _SAMPLE_RATE_HZ)

    assert np.array_equal(first_chroma, second_chroma)
    assert np.array_equal(first_times_seconds, second_times_seconds)


def test_normalize_chroma_leaves_zero_columns_at_zero() -> None:
    chroma = np.zeros((PITCH_CLASS_COUNT, 3), dtype=np.float32)
    chroma[:, 1] = _pitch_class_vector([0, 4, 7])

    normalized = normalize_chroma(chroma)

    np.testing.assert_allclose(
        _column_norms(normalized),
        np.array([0.0, 1.0, 0.0]),
        rtol=0.0,
        atol=1e-6,
    )
    assert not bool(np.any(np.isnan(normalized)))


def test_aggregate_chroma_by_beat_returns_one_vector_per_beat() -> None:
    chroma = _sustained_chroma(24)
    frame_times_seconds = np.arange(24, dtype=np.float64) * 0.1
    beat_times_seconds = np.array([0.0, 0.5, 1.0, 1.5], dtype=np.float64)

    beat_chroma = aggregate_chroma_by_beat(
        chroma,
        frame_times_seconds,
        beat_times_seconds,
    )

    assert beat_chroma.shape == (PITCH_CLASS_COUNT, beat_times_seconds.size)
    assert beat_chroma.dtype == np.dtype(np.float32)
    np.testing.assert_allclose(
        _column_norms(beat_chroma),
        np.ones(beat_times_seconds.size),
        rtol=0.0,
        atol=1e-6,
    )
    for beat_index in range(beat_times_seconds.size):
        assert _strongest_pitch_classes(beat_chroma[:, beat_index]) == {0, 4, 7}


def test_aggregate_chroma_by_beat_is_zero_when_no_pitch_class_persists() -> None:
    # Every frame in the span sounds a different single pitch class, so no pitch
    # class is present in a majority of frames and the median has no content to
    # report. A zero vector scores zero against every chord template, which is a
    # usable "no chord" signal downstream, unlike a NaN.
    chroma = _ramp_chroma(4)
    frame_times_seconds = np.arange(4, dtype=np.float64) * 0.1
    beat_times_seconds = np.array([0.0], dtype=np.float64)

    beat_chroma = aggregate_chroma_by_beat(
        chroma,
        frame_times_seconds,
        beat_times_seconds,
    )

    assert not bool(np.any(np.isnan(beat_chroma)))
    np.testing.assert_allclose(
        beat_chroma,
        np.zeros_like(beat_chroma),
        rtol=0.0,
        atol=0.0,
    )


def test_aggregate_chroma_by_beat_handles_span_with_one_frame() -> None:
    chroma = _ramp_chroma(4)
    frame_times_seconds = np.array([0.0, 0.1, 0.2, 0.3], dtype=np.float64)
    beat_times_seconds = np.array([0.1, 0.2], dtype=np.float64)

    beat_chroma = aggregate_chroma_by_beat(
        chroma,
        frame_times_seconds,
        beat_times_seconds,
    )

    np.testing.assert_allclose(
        beat_chroma[:, 0],
        _pitch_class_vector([1]),
        rtol=0.0,
        atol=1e-6,
    )


def test_aggregate_chroma_by_beat_handles_span_with_no_frames() -> None:
    chroma = _ramp_chroma(3)
    frame_times_seconds = np.array([0.0, 0.1, 0.2], dtype=np.float64)
    beat_times_seconds = np.array([0.11, 0.12, 0.13], dtype=np.float64)

    beat_chroma = aggregate_chroma_by_beat(
        chroma,
        frame_times_seconds,
        beat_times_seconds,
    )

    assert not bool(np.any(np.isnan(beat_chroma)))
    np.testing.assert_allclose(
        _column_norms(beat_chroma),
        np.ones(beat_times_seconds.size),
        rtol=0.0,
        atol=1e-6,
    )
    # The first two spans hold no frame and fall back to the frame at 0.1 s;
    # the open-ended final span contains the frame at 0.2 s.
    expected = np.stack(
        [
            _pitch_class_vector([1]),
            _pitch_class_vector([1]),
            _pitch_class_vector([2]),
        ],
        axis=1,
    )
    np.testing.assert_allclose(beat_chroma, expected, rtol=0.0, atol=1e-6)


def test_aggregate_chroma_by_beat_final_span_reaches_the_last_frame() -> None:
    chroma = np.zeros((PITCH_CLASS_COUNT, 4), dtype=np.float32)
    chroma[0, :2] = 1.0
    chroma[7, 2:] = 1.0
    frame_times_seconds = np.array([0.0, 0.1, 0.2, 0.3], dtype=np.float64)
    beat_times_seconds = np.array([0.0, 0.2], dtype=np.float64)

    beat_chroma = aggregate_chroma_by_beat(
        chroma,
        frame_times_seconds,
        beat_times_seconds,
    )

    np.testing.assert_allclose(
        beat_chroma[:, 1],
        _pitch_class_vector([7]),
        rtol=0.0,
        atol=1e-6,
    )


def test_aggregate_chroma_by_beat_median_resists_an_outlying_frame() -> None:
    chroma = np.zeros((PITCH_CLASS_COUNT, 5), dtype=np.float32)
    chroma[0, :] = 1.0
    # A note attack spreads broadband energy over a single frame.
    chroma[:, 2] = 1.0
    frame_times_seconds = np.arange(5, dtype=np.float64) * 0.1
    beat_times_seconds = np.array([0.0], dtype=np.float64)

    median_chroma = aggregate_chroma_by_beat(
        chroma,
        frame_times_seconds,
        beat_times_seconds,
        aggregation=ChromaAggregation.MEDIAN,
    )
    mean_chroma = aggregate_chroma_by_beat(
        chroma,
        frame_times_seconds,
        beat_times_seconds,
        aggregation=ChromaAggregation.MEAN,
    )

    clean_vector = _pitch_class_vector([0])
    np.testing.assert_allclose(
        median_chroma[:, 0],
        clean_vector,
        rtol=0.0,
        atol=1e-6,
    )
    median_similarity = float(np.dot(median_chroma[:, 0], clean_vector))
    mean_similarity = float(np.dot(mean_chroma[:, 0], clean_vector))
    assert median_similarity > mean_similarity


def test_aggregate_chroma_by_beat_mean_matches_a_manual_mean() -> None:
    chroma = np.zeros((PITCH_CLASS_COUNT, 3), dtype=np.float32)
    chroma[0, :] = np.array([1.0, 0.5, 0.0], dtype=np.float32)
    chroma[4, :] = np.array([0.0, 0.5, 1.0], dtype=np.float32)
    frame_times_seconds = np.array([0.0, 0.1, 0.2], dtype=np.float64)
    beat_times_seconds = np.array([0.0], dtype=np.float64)

    beat_chroma = aggregate_chroma_by_beat(
        chroma,
        frame_times_seconds,
        beat_times_seconds,
        aggregation=ChromaAggregation.MEAN,
    )

    expected = np.mean(chroma.astype(np.float64), axis=1)
    expected = expected / np.linalg.norm(expected)
    np.testing.assert_allclose(
        beat_chroma[:, 0],
        expected,
        rtol=0.0,
        atol=1e-6,
    )


def test_aggregate_chroma_by_beat_tracks_a_chord_change() -> None:
    samples = np.concatenate(
        [
            _triad(_C_MAJOR_HZ, duration_seconds=1.5),
            _triad(_F_MAJOR_HZ, duration_seconds=1.5),
        ]
    )
    beat_times_seconds = np.array([0.0, 1.5], dtype=np.float64)

    chroma, frame_times_seconds = compute_chroma(samples, _SAMPLE_RATE_HZ)
    beat_chroma = aggregate_chroma_by_beat(
        chroma,
        frame_times_seconds,
        beat_times_seconds,
    )

    assert beat_chroma.shape == (PITCH_CLASS_COUNT, 2)
    assert _strongest_pitch_classes(beat_chroma[:, 0]) == {0, 4, 7}
    assert _strongest_pitch_classes(beat_chroma[:, 1]) == {5, 9, 0}


@pytest.mark.parametrize("sample_rate_hz", [0, -1])
def test_compute_chroma_rejects_nonpositive_sample_rate(sample_rate_hz: int) -> None:
    samples = np.zeros(1_000, dtype=np.float32)

    with pytest.raises(ValueError, match="sample_rate_hz must be positive"):
        compute_chroma(samples, sample_rate_hz)


@pytest.mark.parametrize("hop_length_samples", [0, -1])
def test_compute_chroma_rejects_nonpositive_hop_length(
    hop_length_samples: int,
) -> None:
    samples = np.zeros(1_000, dtype=np.float32)

    with pytest.raises(ValueError, match="hop_length_samples must be positive"):
        compute_chroma(
            samples,
            _SAMPLE_RATE_HZ,
            hop_length_samples=hop_length_samples,
        )


@pytest.mark.parametrize(
    "samples, expected_message",
    [
        (np.zeros((10, 2), dtype=np.float32), "one-dimensional"),
        (np.array([], dtype=np.float32), "must not be empty"),
        (np.array([np.nan], dtype=np.float32), "only finite"),
        (np.array([np.inf], dtype=np.float32), "only finite"),
    ],
)
def test_compute_chroma_rejects_unusable_audio(
    samples: npt.NDArray[np.float32],
    expected_message: str,
) -> None:
    with pytest.raises(ValueError, match=expected_message):
        compute_chroma(samples, _SAMPLE_RATE_HZ)


@pytest.mark.parametrize(
    "chroma, expected_message",
    [
        (np.zeros(PITCH_CLASS_COUNT, dtype=np.float32), "shape"),
        (np.zeros((11, 4), dtype=np.float32), "shape"),
        (np.zeros((PITCH_CLASS_COUNT, 0), dtype=np.float32), "at least one frame"),
        (np.full((PITCH_CLASS_COUNT, 2), np.nan, dtype=np.float32), "only finite"),
    ],
)
def test_normalize_chroma_rejects_unusable_matrices(
    chroma: npt.NDArray[np.float32],
    expected_message: str,
) -> None:
    with pytest.raises(ValueError, match=expected_message):
        normalize_chroma(chroma)


@pytest.mark.parametrize(
    "frame_times_seconds, expected_message",
    [
        (np.array([0.0, 0.1], dtype=np.float64), "one timestamp per chroma frame"),
        (np.zeros((3, 1), dtype=np.float64), "one-dimensional"),
        (np.array([0.0, np.nan, 0.2], dtype=np.float64), "only finite"),
        (np.array([0.0, 0.2, 0.1], dtype=np.float64), "strictly increasing"),
        (np.array([0.0, 0.1, 0.1], dtype=np.float64), "strictly increasing"),
    ],
)
def test_aggregate_chroma_by_beat_rejects_unusable_frame_times(
    frame_times_seconds: npt.NDArray[np.float64],
    expected_message: str,
) -> None:
    chroma = _ramp_chroma(3)
    beat_times_seconds = np.array([0.0], dtype=np.float64)

    with pytest.raises(ValueError, match=expected_message):
        aggregate_chroma_by_beat(chroma, frame_times_seconds, beat_times_seconds)


@pytest.mark.parametrize(
    "beat_times_seconds, expected_message",
    [
        (np.array([], dtype=np.float64), "must not be empty"),
        (np.zeros((2, 1), dtype=np.float64), "one-dimensional"),
        (np.array([0.0, np.inf], dtype=np.float64), "only finite"),
        (np.array([-0.1, 0.2], dtype=np.float64), "nonnegative"),
        (np.array([0.2, 0.1], dtype=np.float64), "strictly increasing"),
        (np.array([0.1, 0.1], dtype=np.float64), "strictly increasing"),
    ],
)
def test_aggregate_chroma_by_beat_rejects_unusable_beat_times(
    beat_times_seconds: npt.NDArray[np.float64],
    expected_message: str,
) -> None:
    chroma = _ramp_chroma(3)
    frame_times_seconds = np.array([0.0, 0.1, 0.2], dtype=np.float64)

    with pytest.raises(ValueError, match=expected_message):
        aggregate_chroma_by_beat(chroma, frame_times_seconds, beat_times_seconds)


def test_aggregate_chroma_by_beat_rejects_unusable_chroma() -> None:
    frame_times_seconds = np.array([0.0, 0.1, 0.2], dtype=np.float64)
    beat_times_seconds = np.array([0.0], dtype=np.float64)

    with pytest.raises(ValueError, match="shape"):
        aggregate_chroma_by_beat(
            np.zeros((11, 3), dtype=np.float32),
            frame_times_seconds,
            beat_times_seconds,
        )
