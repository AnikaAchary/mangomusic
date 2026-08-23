# Audio ingestion

`mangomusic.audio` turns an audio file on disk into the one signal
representation the rest of the pipeline accepts: mono, `float32`, at a sample
rate you choose.

Decoding is delegated to [FFmpeg](https://ffmpeg.org/), which must be available
on `PATH`. Any format FFmpeg can read is accepted.

## Public API

```python
from pathlib import Path

from mangomusic.audio import load_audio

samples, sample_rate_hz = load_audio(Path("song.mp3"), 22_050)
```

`load_audio(input_path, target_sample_rate_hz, *, start_time_seconds=0.0,
stop_time_seconds=None)` returns a `(samples, sample_rate_hz)` pair. The returned
sample rate is the one you asked for — it is returned so downstream calls can
take it from the same place rather than repeating the literal.

Decode a section instead of the whole file by passing a time range in seconds.
`start_time_seconds` is inclusive, `stop_time_seconds` exclusive:

```python
samples, sample_rate_hz = load_audio(
    Path("song.mp3"),
    22_050,
    start_time_seconds=30.0,
    stop_time_seconds=45.0,
)
```

## Conventions and units

- **Shape** — `(sample_count,)`. The first audio stream of the file is selected
  and downmixed to mono, so the array is always one-dimensional.
- **Sample rate** — hertz, an integer, and must be positive. Every downstream
  function takes this value alongside the samples, because the sample rate is
  what converts sample counts into seconds.
- **Amplitude** — unitless `float32` in `[-1.0, 1.0]`. A decode whose peak
  exceeds `1.0` is scaled down by its peak, so the range holds for every input.
- **Mutability** — the array is writable, so callers may process it in place.
- **Time range** — seconds. Both bounds must be finite, `start_time_seconds`
  nonnegative, and `stop_time_seconds` strictly greater than the start.

## Errors

Both exceptions come from `mangomusic.errors`, where `AudioDecodeError`
subclasses `MangoMusicError`, the base class for expected MangoMusic failures.

- `ValueError` — the request itself is invalid: a non-positive sample rate or an
  unusable time range. Raised before FFmpeg is invoked.
- `AudioDecodeError` — the input could not be turned into a usable signal:
  a missing or unreadable file, FFmpeg not startable, a nonzero FFmpeg exit, or
  output that does not parse into finite samples.

## Limitations

- FFmpeg must be installed separately; it is not a Python dependency.
- Decoding is a subprocess round trip through a pipe, so whole-file decodes of
  long recordings hold the full signal in memory. Use a time range to work
  section by section.

## Next stage

[Rhythm analysis](rhythm.md) consumes `samples` and `sample_rate_hz` directly.
