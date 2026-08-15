"""Audio decoding and canonicalization."""

import math
import shutil
import subprocess
from pathlib import Path

import numpy as np
import numpy.typing as npt

from mangomusic.errors import AudioDecodeError

_FFMPEG_SAMPLE_DTYPE = np.dtype(np.float32)
_FFMPEG_ERROR_LIMIT = 1_000


def load_audio(
    input_path: Path,
    target_sample_rate_hz: int,
    *,
    start_time_seconds: float = 0.0,
    stop_time_seconds: float | None = None,
) -> tuple[npt.NDArray[np.float32], int]:
    """Load an audio file as mono, finite ``float32`` samples.

    Args:
        input_path: Path to a local audio file.
        target_sample_rate_hz: Output sample rate in hertz. Must be positive.
        start_time_seconds: Inclusive section start time in seconds. Must be
            finite and nonnegative.
        stop_time_seconds: Exclusive section stop time in seconds, or ``None``
            to decode through the end of the file. Must be finite and greater
            than ``start_time_seconds``.

    Returns:
        A writable sample array with shape ``(sample_count,)`` and amplitudes in
        ``[-1.0, 1.0]``, plus the output sample rate in hertz.

    Raises:
        ValueError: If ``target_sample_rate_hz`` or the requested time range is
            invalid.
        AudioDecodeError: If the input or FFmpeg output cannot produce a usable
            audio signal.
    """
    if target_sample_rate_hz <= 0:
        raise ValueError("target_sample_rate_hz must be positive")
    _validate_time_range(start_time_seconds, stop_time_seconds)

    _validate_input_file(input_path)
    ffmpeg_path = _find_ffmpeg()
    command = _build_ffmpeg_command(
        ffmpeg_path,
        input_path,
        target_sample_rate_hz,
        start_time_seconds,
        stop_time_seconds,
    )

    try:
        result = _run_ffmpeg(command)
    except OSError as exc:
        raise AudioDecodeError(f"FFmpeg could not be started: {exc}") from exc

    if result.returncode != 0:
        raise AudioDecodeError(
            _format_decode_error(input_path, result.returncode, result.stderr)
        )

    samples = _parse_samples(input_path, result.stdout)
    peak_amplitude = float(np.max(np.abs(samples)))
    if peak_amplitude > 1.0:
        samples /= np.float32(peak_amplitude)
        np.clip(samples, -1.0, 1.0, out=samples)

    return samples, target_sample_rate_hz


def _validate_time_range(
    start_time_seconds: float,
    stop_time_seconds: float | None,
) -> None:
    if not math.isfinite(start_time_seconds) or start_time_seconds < 0.0:
        raise ValueError("start_time_seconds must be finite and nonnegative")
    if stop_time_seconds is None:
        return
    if not math.isfinite(stop_time_seconds):
        raise ValueError("stop_time_seconds must be finite")
    if stop_time_seconds <= start_time_seconds:
        raise ValueError("stop_time_seconds must be greater than start_time_seconds")


def _validate_input_file(input_path: Path) -> None:
    if not input_path.exists():
        raise AudioDecodeError(f"Audio file does not exist: {input_path}")
    if not input_path.is_file():
        raise AudioDecodeError(f"Audio input is not a file: {input_path}")

    try:
        with input_path.open("rb"):
            pass
    except OSError as exc:
        raise AudioDecodeError(f"Audio file is not readable: {input_path}") from exc


def _find_ffmpeg() -> Path:
    executable = shutil.which("ffmpeg")
    if executable is None:
        raise AudioDecodeError(
            "FFmpeg was not found. Install it and ensure it is available on PATH."
        )
    return Path(executable)


def _build_ffmpeg_command(
    ffmpeg_path: Path,
    input_path: Path,
    target_sample_rate_hz: int,
    start_time_seconds: float,
    stop_time_seconds: float | None,
) -> list[str]:
    command = [
        str(ffmpeg_path),
        "-hide_banner",
        "-loglevel",
        "error",
        "-nostdin",
        "-i",
        str(input_path),
    ]
    if start_time_seconds > 0.0:
        command.extend(["-ss", _format_time_seconds(start_time_seconds)])
    if stop_time_seconds is not None:
        duration_seconds = stop_time_seconds - start_time_seconds
        command.extend(["-t", _format_time_seconds(duration_seconds)])
    command.extend(
        [
            "-map",
            "0:a:0",
            "-vn",
            "-sn",
            "-dn",
            "-ac",
            "1",
            "-ar",
            str(target_sample_rate_hz),
            "-f",
            "f32le",
            "-codec:a",
            "pcm_f32le",
            "pipe:1",
        ]
    )
    return command


def _format_time_seconds(time_seconds: float) -> str:
    return format(time_seconds, ".12g")


def _run_ffmpeg(command: list[str]) -> subprocess.CompletedProcess[bytes]:
    return subprocess.run(
        command,
        capture_output=True,
        check=False,
    )


def _format_decode_error(
    input_path: Path,
    return_code: int,
    stderr: bytes,
) -> str:
    details = stderr.decode("utf-8", errors="replace").strip()
    if len(details) > _FFMPEG_ERROR_LIMIT:
        details = f"...{details[-_FFMPEG_ERROR_LIMIT:]}"
    if not details:
        details = f"FFmpeg exited with status {return_code}"
    return f"Could not decode audio file {input_path}: {details}"


def _parse_samples(
    input_path: Path,
    decoded_bytes: bytes,
) -> npt.NDArray[np.float32]:
    if not decoded_bytes:
        raise AudioDecodeError(f"Decoded audio is empty: {input_path}")
    if len(decoded_bytes) % _FFMPEG_SAMPLE_DTYPE.itemsize != 0:
        raise AudioDecodeError(f"FFmpeg returned malformed audio data: {input_path}")

    samples: npt.NDArray[np.float32] = np.frombuffer(
        decoded_bytes,
        dtype=_FFMPEG_SAMPLE_DTYPE,
    ).astype(np.float32, copy=True)
    if not bool(np.all(np.isfinite(samples))):
        raise AudioDecodeError(
            f"Decoded audio contains non-finite samples: {input_path}"
        )

    return samples
