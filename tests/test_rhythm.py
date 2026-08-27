"""Tests for onset extraction and rhythm analysis."""

import numpy as np
import numpy.typing as npt
import pytest
from pydantic import ValidationError

from mangomusic.rhythm import (
    BeatEvent,
    RhythmAnalysis,
    analyze_rhythm,
    compute_onset_strength,
)


def _click_track(
    sample_rate_hz: int,
    bpm: float,
    duration_seconds: float,
    first_beat_seconds: float,
) -> tuple[npt.NDArray[np.float32], npt.NDArray[np.float64]]:
    sample_count = round(sample_rate_hz * duration_seconds)
    samples = np.zeros(sample_count, dtype=np.float32)
    beat_interval_seconds = 60.0 / bpm
    beat_times_seconds = np.arange(
        first_beat_seconds,
        duration_seconds - 0.1,
        beat_interval_seconds,
        dtype=np.float64,
    )

    click_sample_count = round(0.02 * sample_rate_hz)
    click_indices = np.arange(click_sample_count, dtype=np.float64)
    click = np.asarray(
        0.9
        * np.exp(-click_indices / (0.003 * sample_rate_hz))
        * np.sin(2.0 * np.pi * 1_000.0 * click_indices / sample_rate_hz),
        dtype=np.float32,
    )
    for beat_time_seconds in beat_times_seconds:
        start = round(float(beat_time_seconds) * sample_rate_hz)
        samples[start : start + click_sample_count] = click

    return samples, beat_times_seconds


def _accented_click_track(
    sample_rate_hz: int,
    bpm: float,
    duration_seconds: float,
    first_beat_seconds: float,
    first_downbeat_index: int,
) -> tuple[npt.NDArray[np.float32], npt.NDArray[np.float64]]:
    samples, beat_times_seconds = _click_track(
        sample_rate_hz,
        bpm,
        duration_seconds,
        first_beat_seconds,
    )
    accent_sample_count = round(0.08 * sample_rate_hz)
    accent_indices = np.arange(accent_sample_count, dtype=np.float64)
    accent = np.asarray(
        0.8
        * np.exp(-accent_indices / (0.02 * sample_rate_hz))
        * np.sin(2.0 * np.pi * 80.0 * accent_indices / sample_rate_hz),
        dtype=np.float32,
    )
    for beat_index in range(first_downbeat_index, beat_times_seconds.size, 4):
        start = round(float(beat_times_seconds[beat_index]) * sample_rate_hz)
        samples[start : start + accent_sample_count] += accent
    np.clip(samples, -1.0, 1.0, out=samples)
    return samples, beat_times_seconds


def _valid_beat() -> BeatEvent:
    return BeatEvent(
        timestamp_seconds=0.5,
        bar_number=1,
        beat_in_bar=1,
        confidence=0.8,
    )


def test_analyze_rhythm_tracks_fixed_bpm_clicks() -> None:
    sample_rate_hz = 22_050
    expected_bpm = 120.0
    samples, expected_times_seconds = _click_track(
        sample_rate_hz,
        expected_bpm,
        duration_seconds=12.0,
        first_beat_seconds=0.5,
    )

    onset_strength, onset_times_seconds = compute_onset_strength(
        samples,
        sample_rate_hz,
    )
    analysis = analyze_rhythm(samples, sample_rate_hz)

    assert onset_strength.dtype == np.dtype(np.float32)
    assert onset_strength.shape == onset_times_seconds.shape
    assert onset_strength.size > 0
    assert bool(np.all(np.diff(onset_times_seconds) > 0.0))
    assert not analysis.is_silent
    assert analysis.bpm is not None
    np.testing.assert_allclose(
        analysis.bpm,
        expected_bpm,
        rtol=0.0,
        atol=1.0,
    )
    assert len(analysis.beats) >= expected_times_seconds.size - 2

    detected_times_seconds = np.array(
        [beat.timestamp_seconds for beat in analysis.beats],
        dtype=np.float64,
    )
    nearest_expected_errors = np.min(
        np.abs(detected_times_seconds[:, np.newaxis] - expected_times_seconds),
        axis=1,
    )
    np.testing.assert_allclose(
        nearest_expected_errors,
        np.zeros_like(nearest_expected_errors),
        rtol=0.0,
        atol=0.03,
    )
    assert bool(np.all(np.diff(detected_times_seconds) > 0.0))
    assert [beat.beat_in_bar for beat in analysis.beats] == [
        index % 4 + 1 for index in range(len(analysis.beats))
    ]
    assert [beat.bar_number for beat in analysis.beats] == [
        index // 4 + 1 for index in range(len(analysis.beats))
    ]
    assert analysis.downbeat_confidence is None
    assert analysis.tempo_confidence > 0.7
    beat_confidences = np.array(
        [beat.confidence for beat in analysis.beats],
        dtype=np.float64,
    )
    assert float(np.mean(beat_confidences)) > 0.7


@pytest.mark.parametrize("first_downbeat_index", [0, 1, 2, 3])
def test_analyze_rhythm_infers_downbeat_phase(first_downbeat_index: int) -> None:
    sample_rate_hz = 22_050
    samples, expected_times_seconds = _accented_click_track(
        sample_rate_hz,
        bpm=120.0,
        duration_seconds=12.0,
        first_beat_seconds=0.5,
        first_downbeat_index=first_downbeat_index,
    )

    analysis = analyze_rhythm(samples, sample_rate_hz)

    assert analysis.downbeat_confidence is not None
    detected_times_seconds = np.array(
        [beat.timestamp_seconds for beat in analysis.beats],
        dtype=np.float64,
    )
    expected_indices = np.argmin(
        np.abs(detected_times_seconds[:, np.newaxis] - expected_times_seconds),
        axis=1,
    )
    expected_beats_in_bar = [
        (int(index) - first_downbeat_index) % 4 + 1 for index in expected_indices
    ]
    assert [beat.beat_in_bar for beat in analysis.beats] == expected_beats_in_bar


def test_analyze_rhythm_numbers_a_leading_partial_bar() -> None:
    sample_rate_hz = 22_050
    samples, _ = _accented_click_track(
        sample_rate_hz,
        bpm=120.0,
        duration_seconds=12.0,
        first_beat_seconds=0.5,
        first_downbeat_index=2,
    )

    analysis = analyze_rhythm(samples, sample_rate_hz)

    assert analysis.downbeat_confidence is not None
    assert analysis.beats[0].bar_number == 1
    for previous, current in zip(analysis.beats, analysis.beats[1:], strict=False):
        expected_increment = int(current.beat_in_bar == 1)
        assert current.bar_number == previous.bar_number + expected_increment


def test_analyze_rhythm_does_not_infer_downbeat_from_short_audio() -> None:
    sample_rate_hz = 22_050
    samples, _ = _accented_click_track(
        sample_rate_hz,
        bpm=120.0,
        duration_seconds=6.0,
        first_beat_seconds=0.5,
        first_downbeat_index=2,
    )

    analysis = analyze_rhythm(samples, sample_rate_hz)

    assert analysis.bpm is not None
    assert len(analysis.beats) < 12
    assert analysis.downbeat_confidence is None
    assert [beat.beat_in_bar for beat in analysis.beats] == [
        index % 4 + 1 for index in range(len(analysis.beats))
    ]


def test_analyze_rhythm_recognizes_silence() -> None:
    sample_rate_hz = 22_050
    samples = np.zeros(sample_rate_hz * 3, dtype=np.float32)

    onset_strength, _ = compute_onset_strength(samples, sample_rate_hz)
    analysis = analyze_rhythm(samples, sample_rate_hz)

    np.testing.assert_allclose(
        onset_strength,
        np.zeros_like(onset_strength),
        rtol=0.0,
        atol=0.0,
    )
    assert analysis.is_silent
    assert analysis.bpm is None
    assert analysis.beats == []
    np.testing.assert_allclose(
        analysis.tempo_confidence,
        0.0,
        rtol=0.0,
        atol=0.0,
    )


def test_analyze_rhythm_recognizes_non_rhythmic_audio() -> None:
    sample_rate_hz = 22_050
    sample_indices = np.arange(sample_rate_hz * 3, dtype=np.float64)
    samples = np.asarray(
        0.2 * np.sin(2.0 * np.pi * 440.0 * sample_indices / sample_rate_hz),
        dtype=np.float32,
    )

    analysis = analyze_rhythm(samples, sample_rate_hz)

    assert not analysis.is_silent
    assert analysis.bpm is None
    assert analysis.beats == []
    np.testing.assert_allclose(
        analysis.tempo_confidence,
        0.0,
        rtol=0.0,
        atol=0.0,
    )


@pytest.mark.parametrize("sample_rate_hz", [0, -1])
def test_rhythm_functions_reject_nonpositive_sample_rate(
    sample_rate_hz: int,
) -> None:
    samples = np.zeros(1_000, dtype=np.float32)

    with pytest.raises(ValueError, match="must be positive"):
        compute_onset_strength(samples, sample_rate_hz)
    with pytest.raises(ValueError, match="must be positive"):
        analyze_rhythm(samples, sample_rate_hz)


@pytest.mark.parametrize(
    "samples, expected_message",
    [
        (np.zeros((10, 2), dtype=np.float32), "one-dimensional"),
        (np.array([], dtype=np.float32), "must not be empty"),
        (np.array([np.nan], dtype=np.float32), "only finite"),
        (np.array([np.inf], dtype=np.float32), "only finite"),
    ],
)
def test_analyze_rhythm_rejects_unusable_audio(
    samples: npt.NDArray[np.float32],
    expected_message: str,
) -> None:
    with pytest.raises(ValueError, match=expected_message):
        analyze_rhythm(samples, 22_050)


@pytest.mark.parametrize(
    "field, value",
    [
        ("timestamp_seconds", -0.1),
        ("bar_number", 0),
        ("beat_in_bar", 0),
        ("beat_in_bar", 5),
        ("confidence", -0.1),
        ("confidence", 1.1),
    ],
)
def test_beat_event_rejects_out_of_range_fields(field: str, value: float) -> None:
    values: dict[str, float | int] = {
        "timestamp_seconds": 0.5,
        "bar_number": 1,
        "beat_in_bar": 1,
        "confidence": 0.8,
    }
    values[field] = value

    with pytest.raises(ValidationError):
        BeatEvent.model_validate(values)


@pytest.mark.parametrize(
    "field, value",
    [
        ("bpm", 0.0),
        ("bpm", -1.0),
        ("tempo_confidence", -0.1),
        ("tempo_confidence", 1.1),
        ("downbeat_confidence", -0.1),
        ("downbeat_confidence", 1.1),
    ],
)
def test_rhythm_analysis_rejects_out_of_range_fields(
    field: str,
    value: float,
) -> None:
    values: dict[str, object] = {
        "bpm": 120.0,
        "tempo_confidence": 0.8,
        "beats": [_valid_beat()],
        "is_silent": False,
    }
    values[field] = value

    with pytest.raises(ValidationError):
        RhythmAnalysis.model_validate(values)


@pytest.mark.parametrize(
    "values",
    [
        {
            "bpm": None,
            "tempo_confidence": 0.5,
            "beats": [],
            "is_silent": False,
        },
        {
            "bpm": None,
            "tempo_confidence": 0.0,
            "downbeat_confidence": 0.5,
            "beats": [],
            "is_silent": False,
        },
        {
            "bpm": None,
            "tempo_confidence": 0.0,
            "beats": [_valid_beat()],
            "is_silent": False,
        },
        {
            "bpm": 120.0,
            "tempo_confidence": 0.5,
            "beats": [_valid_beat()],
            "is_silent": True,
        },
        {
            "bpm": 120.0,
            "tempo_confidence": 0.5,
            "beats": [],
            "is_silent": False,
        },
    ],
)
def test_rhythm_analysis_rejects_inconsistent_results(
    values: dict[str, object],
) -> None:
    with pytest.raises(ValidationError):
        RhythmAnalysis.model_validate(values)
