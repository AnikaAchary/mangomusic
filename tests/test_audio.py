"""Tests for the audio ingestion boundary."""

import subprocess
from pathlib import Path

import numpy as np
import pytest

from mangomusic import audio
from mangomusic.errors import AudioDecodeError


def _audio_file(tmp_path: Path) -> Path:
    input_path = tmp_path / "song with spaces.wav"
    input_path.write_bytes(b"placeholder")
    return input_path


def _use_ffmpeg_result(
    monkeypatch: pytest.MonkeyPatch,
    result: subprocess.CompletedProcess[bytes],
) -> list[list[str]]:
    commands: list[list[str]] = []

    def find_ffmpeg() -> Path:
        return Path("ffmpeg")

    def run_ffmpeg(command: list[str]) -> subprocess.CompletedProcess[bytes]:
        commands.append(command)
        return result

    monkeypatch.setattr(audio, "_find_ffmpeg", find_ffmpeg)
    monkeypatch.setattr(audio, "_run_ffmpeg", run_ffmpeg)
    return commands


@pytest.mark.parametrize("sample_rate_hz", [0, -1])
def test_load_audio_rejects_nonpositive_sample_rate(
    tmp_path: Path,
    sample_rate_hz: int,
) -> None:
    with pytest.raises(ValueError, match="must be positive"):
        audio.load_audio(tmp_path / "missing.wav", sample_rate_hz)


def test_load_audio_rejects_missing_input(tmp_path: Path) -> None:
    with pytest.raises(AudioDecodeError, match="does not exist"):
        audio.load_audio(tmp_path / "missing.wav", 22_050)


def test_load_audio_rejects_directory(tmp_path: Path) -> None:
    with pytest.raises(AudioDecodeError, match="not a file"):
        audio.load_audio(tmp_path, 22_050)


def test_load_audio_reports_missing_ffmpeg(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    input_path = _audio_file(tmp_path)

    def missing_executable(_command: str) -> None:
        return None

    monkeypatch.setattr(audio.shutil, "which", missing_executable)

    with pytest.raises(AudioDecodeError, match="FFmpeg was not found"):
        audio.load_audio(input_path, 22_050)


def test_load_audio_returns_normalized_writable_float32_samples(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    input_path = _audio_file(tmp_path)
    decoded = np.array([0.0, 0.5, -2.0, 1.0], dtype=np.float32)
    result = subprocess.CompletedProcess(
        args=["ffmpeg"],
        returncode=0,
        stdout=decoded.tobytes(),
        stderr=b"",
    )
    commands = _use_ffmpeg_result(monkeypatch, result)

    samples, sample_rate_hz = audio.load_audio(input_path, 22_050)

    np.testing.assert_allclose(
        samples,
        np.array([0.0, 0.25, -1.0, 0.5], dtype=np.float32),
        rtol=0.0,
        atol=1e-7,
    )
    assert samples.dtype == np.dtype(np.float32)
    assert samples.flags.c_contiguous
    assert samples.flags.writeable
    assert sample_rate_hz == 22_050
    assert str(input_path) in commands[0]


def test_load_audio_does_not_amplify_quiet_audio(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    input_path = _audio_file(tmp_path)
    decoded = np.array([-0.1, 0.0, 0.2], dtype=np.float32)
    result = subprocess.CompletedProcess(
        args=["ffmpeg"],
        returncode=0,
        stdout=decoded.tobytes(),
        stderr=b"",
    )
    _use_ffmpeg_result(monkeypatch, result)

    samples, _ = audio.load_audio(input_path, 44_100)

    np.testing.assert_allclose(samples, decoded, rtol=0.0, atol=1e-7)


def test_load_audio_reports_ffmpeg_failure(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    input_path = _audio_file(tmp_path)
    result = subprocess.CompletedProcess(
        args=["ffmpeg"],
        returncode=1,
        stdout=b"",
        stderr=b"Invalid data found when processing input",
    )
    _use_ffmpeg_result(monkeypatch, result)

    with pytest.raises(AudioDecodeError, match="Invalid data found"):
        audio.load_audio(input_path, 22_050)


@pytest.mark.parametrize(
    "decoded_bytes, expected_message",
    [
        (b"", "Decoded audio is empty"),
        (b"abc", "malformed audio data"),
        (np.array([np.nan], dtype=np.float32).tobytes(), "non-finite samples"),
        (np.array([np.inf], dtype=np.float32).tobytes(), "non-finite samples"),
    ],
)
def test_load_audio_rejects_unusable_decoder_output(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    decoded_bytes: bytes,
    expected_message: str,
) -> None:
    input_path = _audio_file(tmp_path)
    result = subprocess.CompletedProcess(
        args=["ffmpeg"],
        returncode=0,
        stdout=decoded_bytes,
        stderr=b"",
    )
    _use_ffmpeg_result(monkeypatch, result)

    with pytest.raises(AudioDecodeError, match=expected_message):
        audio.load_audio(input_path, 22_050)
