"""Tests for the audio ingestion boundary."""

import subprocess
import wave
from pathlib import Path

import numpy as np
import numpy.typing as npt
import pytest

from mangomusic import audio
from mangomusic.errors import AudioDecodeError

_PCM16_SCALE = np.float32(32_768.0)


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


def _sine_wave(
    sample_rate_hz: int,
    duration_seconds: float,
    frequency_hz: float,
    amplitude: float = 0.25,
) -> npt.NDArray[np.float32]:
    sample_count = round(sample_rate_hz * duration_seconds)
    sample_indices = np.arange(sample_count, dtype=np.float64)
    phase = 2.0 * np.pi * frequency_hz * sample_indices / sample_rate_hz
    return np.asarray(amplitude * np.sin(phase), dtype=np.float32)


def _write_pcm16_wav(
    path: Path,
    samples: npt.NDArray[np.float32],
    sample_rate_hz: int,
) -> npt.NDArray[np.int16]:
    if samples.ndim == 1:
        channel_count = 1
    elif samples.ndim == 2:
        channel_count = samples.shape[1]
    else:
        raise ValueError("WAV samples must have one or two dimensions")

    pcm_samples: npt.NDArray[np.int16] = np.rint(
        np.clip(samples, -1.0, 1.0) * np.iinfo(np.int16).max
    ).astype(np.int16)
    with wave.open(str(path), "wb") as wav_file:
        wav_file.setnchannels(channel_count)
        wav_file.setsampwidth(np.dtype(np.int16).itemsize)
        wav_file.setframerate(sample_rate_hz)
        wav_file.writeframes(pcm_samples.tobytes())

    return pcm_samples


def _root_mean_square(samples: npt.NDArray[np.float32]) -> float:
    values = samples.astype(np.float64)
    return float(np.sqrt(np.mean(values * values)))


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


@pytest.mark.parametrize("start_time_seconds", [-1.0, float("nan"), float("inf")])
def test_load_audio_rejects_invalid_start_time(
    tmp_path: Path,
    start_time_seconds: float,
) -> None:
    with pytest.raises(ValueError, match="start_time_seconds"):
        audio.load_audio(
            tmp_path / "missing.wav",
            22_050,
            start_time_seconds=start_time_seconds,
        )


@pytest.mark.parametrize("stop_time_seconds", [float("nan"), float("inf")])
def test_load_audio_rejects_nonfinite_stop_time(
    tmp_path: Path,
    stop_time_seconds: float,
) -> None:
    with pytest.raises(ValueError, match="stop_time_seconds must be finite"):
        audio.load_audio(
            tmp_path / "missing.wav",
            22_050,
            stop_time_seconds=stop_time_seconds,
        )


@pytest.mark.parametrize("stop_time_seconds", [-1.0, 0.0, 4.0])
def test_load_audio_rejects_stop_time_not_after_start(
    tmp_path: Path,
    stop_time_seconds: float,
) -> None:
    with pytest.raises(ValueError, match="must be greater"):
        audio.load_audio(
            tmp_path / "missing.wav",
            22_050,
            start_time_seconds=4.0,
            stop_time_seconds=stop_time_seconds,
        )


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


def test_load_audio_passes_requested_time_range_to_ffmpeg(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    input_path = _audio_file(tmp_path)
    decoded = np.array([0.1], dtype=np.float32)
    result = subprocess.CompletedProcess(
        args=["ffmpeg"],
        returncode=0,
        stdout=decoded.tobytes(),
        stderr=b"",
    )
    commands = _use_ffmpeg_result(monkeypatch, result)

    audio.load_audio(
        input_path,
        22_050,
        start_time_seconds=12.5,
        stop_time_seconds=15.75,
    )

    command = commands[0]
    assert command[command.index("-ss") + 1] == "12.5"
    assert command[command.index("-t") + 1] == "3.25"
    assert command.index("-ss") > command.index("-i")


def test_load_audio_omits_time_range_options_by_default(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    input_path = _audio_file(tmp_path)
    decoded = np.array([0.1], dtype=np.float32)
    result = subprocess.CompletedProcess(
        args=["ffmpeg"],
        returncode=0,
        stdout=decoded.tobytes(),
        stderr=b"",
    )
    commands = _use_ffmpeg_result(monkeypatch, result)

    audio.load_audio(input_path, 22_050)

    assert "-ss" not in commands[0]
    assert "-t" not in commands[0]


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


def test_load_audio_decodes_generated_mono_wav(tmp_path: Path) -> None:
    sample_rate_hz = 22_050
    input_path = tmp_path / "generated mono audio.wav"
    source = _sine_wave(sample_rate_hz, 1.0, 440.0)
    pcm_source = _write_pcm16_wav(input_path, source, sample_rate_hz)

    samples, actual_sample_rate_hz = audio.load_audio(input_path, sample_rate_hz)

    expected = pcm_source.astype(np.float32) / _PCM16_SCALE
    assert samples.shape == (sample_rate_hz,)
    assert samples.dtype == np.dtype(np.float32)
    assert actual_sample_rate_hz == sample_rate_hz
    assert bool(np.all(np.isfinite(samples)))
    assert samples.flags.writeable
    np.testing.assert_allclose(
        samples,
        expected,
        rtol=0.0,
        atol=1.0 / float(_PCM16_SCALE),
    )


def test_load_audio_downmixes_generated_stereo_wav(tmp_path: Path) -> None:
    sample_rate_hz = 22_050
    half_sample_count = sample_rate_hz // 2
    tone = _sine_wave(sample_rate_hz, 0.5, 440.0)
    stereo = np.zeros((sample_rate_hz, 2), dtype=np.float32)
    stereo[:half_sample_count, 0] = tone
    stereo[half_sample_count:, 1] = tone
    input_path = tmp_path / "generated stereo audio.wav"
    _write_pcm16_wav(input_path, stereo, sample_rate_hz)

    samples, _ = audio.load_audio(input_path, sample_rate_hz)

    margin_samples = 256
    left_only_rms = _root_mean_square(
        samples[margin_samples : half_sample_count - margin_samples]
    )
    right_only_rms = _root_mean_square(
        samples[half_sample_count + margin_samples : -margin_samples]
    )
    assert samples.shape == (sample_rate_hz,)
    assert left_only_rms > 0.01
    assert right_only_rms > 0.01
    np.testing.assert_allclose(
        left_only_rms,
        right_only_rms,
        rtol=0.01,
        atol=1e-5,
    )


def test_load_audio_resamples_generated_wav(tmp_path: Path) -> None:
    source_sample_rate_hz = 48_000
    target_sample_rate_hz = 22_050
    frequency_hz = 440.0
    source = _sine_wave(source_sample_rate_hz, 1.0, frequency_hz)
    input_path = tmp_path / "generated resampled audio.wav"
    _write_pcm16_wav(input_path, source, source_sample_rate_hz)

    samples, actual_sample_rate_hz = audio.load_audio(
        input_path,
        target_sample_rate_hz,
    )

    assert actual_sample_rate_hz == target_sample_rate_hz
    assert abs(samples.size - target_sample_rate_hz) <= 1
    duration_seconds = samples.size / actual_sample_rate_hz
    np.testing.assert_allclose(
        duration_seconds,
        1.0,
        rtol=0.0,
        atol=1.0 / target_sample_rate_hz,
    )

    spectrum = np.abs(np.fft.rfft(samples.astype(np.float64)))
    peak_index = int(np.argmax(spectrum[1:])) + 1
    frequencies_hz = np.fft.rfftfreq(
        samples.size,
        d=1.0 / actual_sample_rate_hz,
    )
    peak_frequency_hz = float(frequencies_hz[peak_index])
    np.testing.assert_allclose(
        peak_frequency_hz,
        frequency_hz,
        rtol=0.0,
        atol=1.0,
    )


def test_load_audio_rejects_corrupt_file_with_real_ffmpeg(tmp_path: Path) -> None:
    input_path = tmp_path / "corrupt audio.wav"
    input_path.write_bytes(b"this is not an audio file")

    with pytest.raises(AudioDecodeError) as error:
        audio.load_audio(input_path, 22_050)

    message = str(error.value)
    assert "Could not decode audio file" in message
    assert str(input_path) in message
