"""Tests for sample-recording lookup and ground-truth manifest parsing."""

import json
from pathlib import Path

import pytest
from pydantic import ValidationError

from mangomusic.recordings import (
    SAMPLES_DIR_ENV_VAR,
    RecordingManifest,
    RecordingWindow,
    recording_path,
    samples_dir,
)

MANIFEST_PATH = Path(__file__).parent / "data" / "recordings.json"


def test_chord_window_expands_to_triad_pitch_classes() -> None:
    window = RecordingWindow(start_seconds=1.0, stop_seconds=2.0, chord="C:maj")

    assert window.expected_pitch_classes == (0, 4, 7)
    assert window.expect_in_tune is True


def test_note_window_expands_to_named_pitch_classes() -> None:
    window = RecordingWindow(start_seconds=1.0, stop_seconds=2.0, notes=("G", "C"))

    assert window.expected_pitch_classes == (0, 7)


@pytest.mark.parametrize(
    "kwargs, expected_message",
    [
        ({"start_seconds": 2.0, "stop_seconds": 1.0, "chord": "C:maj"}, "greater than"),
        ({"start_seconds": 1.0, "stop_seconds": 1.0, "chord": "C:maj"}, "greater than"),
        (
            {"start_seconds": -1.0, "stop_seconds": 2.0, "chord": "C:maj"},
            "greater_than",
        ),
        ({"start_seconds": 1.0, "stop_seconds": 2.0}, "exactly one"),
        (
            {
                "start_seconds": 1.0,
                "stop_seconds": 2.0,
                "chord": "C:maj",
                "notes": ("C",),
            },
            "exactly one",
        ),
        (
            {"start_seconds": 1.0, "stop_seconds": 2.0, "chord": "C:sus4"},
            "recognized chord",
        ),
        ({"start_seconds": 1.0, "stop_seconds": 2.0, "notes": ()}, "must not be empty"),
        (
            {"start_seconds": 1.0, "stop_seconds": 2.0, "notes": ("H",)},
            "pitch classes",
        ),
    ],
)
def test_recording_window_rejects_unusable_input(
    kwargs: dict[str, object],
    expected_message: str,
) -> None:
    with pytest.raises(ValidationError, match=expected_message):
        RecordingWindow(**kwargs)  # pyright: ignore[reportArgumentType]


def test_manifest_round_trips_through_json(tmp_path: Path) -> None:
    manifest = RecordingManifest(
        recordings={
            "take.wav": [
                RecordingWindow(
                    start_seconds=0.0,
                    stop_seconds=1.0,
                    chord="A:min",
                    expect_in_tune=False,
                )
            ]
        }
    )
    manifest_path = tmp_path / "recordings.json"
    manifest_path.write_text(manifest.model_dump_json())

    assert RecordingManifest.from_path(manifest_path) == manifest


def test_manifest_rejects_unusable_json(tmp_path: Path) -> None:
    manifest_path = tmp_path / "recordings.json"
    manifest_path.write_text(
        json.dumps(
            {"recordings": {"take.wav": [{"start_seconds": 0.0, "chord": "C:maj"}]}}
        )
    )

    with pytest.raises(ValidationError):
        RecordingManifest.from_path(manifest_path)


def test_committed_manifest_is_valid() -> None:
    manifest = RecordingManifest.from_path(MANIFEST_PATH)

    assert manifest.recordings
    for windows in manifest.recordings.values():
        assert windows
        for window in windows:
            assert 1 <= len(window.expected_pitch_classes) <= 3


def test_samples_dir_is_absent_when_configured_path_does_not_exist(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setenv(SAMPLES_DIR_ENV_VAR, str(tmp_path / "missing"))

    assert samples_dir() is None
    assert recording_path("take.wav") is None


def test_recording_path_finds_a_file_in_the_configured_directory(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setenv(SAMPLES_DIR_ENV_VAR, str(tmp_path))
    (tmp_path / "take.wav").write_bytes(b"")

    assert samples_dir() == tmp_path
    assert recording_path("take.wav") == tmp_path / "take.wav"
    assert recording_path("absent.wav") is None
