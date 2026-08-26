"""Checks that the sample guitar recordings contain the harmony they claim to.

These tests read real audio, which is not committed, so they skip unless the
recordings are present. See ``docs/contributing.md`` for how to point
``MANGOMUSIC_SAMPLES_DIR`` at them.

Two questions are asked of every annotated window. The 12-bin chromagram answers
whether the expected notes sound at all; the 36-bin view answers whether their
energy sits on the semitone center or has drifted off it, which is the only way
to tell a flat note from a different note.
"""

from pathlib import Path

import numpy as np
import numpy.typing as npt
import pytest

from mangomusic.audio import load_audio
from mangomusic.chroma import (
    BINS_PER_SEMITONE,
    PITCH_CLASS_NAMES,
    compute_chroma,
    compute_semitone_chroma,
)
from mangomusic.recordings import RecordingManifest, RecordingWindow, recording_path

_SAMPLE_RATE_HZ = 22_050

# The weakest annotated window separates its last expected pitch class from the
# first unexpected one by a factor of 1.65, and the strongest by 12.3. Requiring
# 1.4 keeps every window comfortably inside while still failing a chromagram in
# which a fourth pitch class competes with the triad.
_PITCH_CLASS_MARGIN = 1.4

# Energy in the centered bin divided by the louder of its flat and sharp
# neighbours. In-tune windows measure 1.57 to 1.77; the deliberately flat take
# measures 0.83 to 1.00. The threshold sits between them with roughly a quarter
# of headroom on each side.
_CENTERED_ENERGY_RATIO = 1.25

_MANIFEST = RecordingManifest.from_path(
    Path(__file__).parent / "data" / "recordings.json"
)
_WINDOWS = [
    pytest.param(file_name, window, id=f"{file_name}@{window.start_seconds:g}s")
    for file_name, windows in _MANIFEST.recordings.items()
    for window in windows
]


def _window_chroma(
    file_name: str,
    window: RecordingWindow,
    *,
    semitone_resolution: bool,
) -> npt.NDArray[np.float64]:
    """Average the chroma of one annotated window across its frames.

    Decoding is limited to the window so that the several seconds of noise floor
    at each end of these recordings, where the chromagram is flat and carries no
    harmony, cannot influence the result.

    Args:
        file_name: Name of the recording within the samples directory.
        window: The annotated span to analyze.
        semitone_resolution: Whether to return the 36-bin view instead of the
            12-bin one.

    Returns:
        Unitless mean energies with shape ``(36,)`` when ``semitone_resolution``
        is set and ``(12,)`` otherwise.
    """
    path = recording_path(file_name)
    if path is None:
        pytest.skip(f"sample recording is not available: {file_name}")

    samples, sample_rate_hz = load_audio(
        path,
        _SAMPLE_RATE_HZ,
        start_time_seconds=window.start_seconds,
        stop_time_seconds=window.stop_seconds,
    )
    compute = compute_semitone_chroma if semitone_resolution else compute_chroma
    chroma, _ = compute(samples, sample_rate_hz)
    return chroma.astype(np.float64).mean(axis=1)


@pytest.mark.parametrize("file_name, window", _WINDOWS)
def test_expected_pitch_classes_dominate_the_chromagram(
    file_name: str,
    window: RecordingWindow,
) -> None:
    summary = _window_chroma(file_name, window, semitone_resolution=False)
    expected = window.expected_pitch_classes

    ranked = np.argsort(summary)[::-1]
    strongest = tuple(sorted(int(index) for index in ranked[: len(expected)]))
    assert strongest == expected, (
        f"{file_name} at {window.start_seconds:g}s: expected "
        f"{[PITCH_CLASS_NAMES[index] for index in expected]}, "
        f"got {[PITCH_CLASS_NAMES[index] for index in strongest]}"
    )

    weakest_expected = float(summary[ranked[len(expected) - 1]])
    strongest_other = float(summary[ranked[len(expected)]])
    assert weakest_expected > _PITCH_CLASS_MARGIN * strongest_other


@pytest.mark.parametrize("file_name, window", _WINDOWS)
def test_spectral_power_sits_in_the_intended_frequency_bins(
    file_name: str,
    window: RecordingWindow,
) -> None:
    summary = _window_chroma(file_name, window, semitone_resolution=True)

    for pitch_class in window.expected_pitch_classes:
        start = BINS_PER_SEMITONE * pitch_class
        flat, centered, sharp = summary[start : start + BINS_PER_SEMITONE]
        ratio = float(centered / max(flat, sharp))
        detail = (
            f"{file_name} at {window.start_seconds:g}s, "
            f"{PITCH_CLASS_NAMES[pitch_class]}: centered/neighbour = {ratio:.2f}"
        )
        if window.expect_in_tune:
            assert ratio > _CENTERED_ENERGY_RATIO, f"{detail} (expected in tune)"
        else:
            assert ratio < _CENTERED_ENERGY_RATIO, f"{detail} (expected out of tune)"
