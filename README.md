# mangomusic

Automatic music transcription: audio in, chord sheet out.

`mangomusic` takes a recording and produces a readable chord chart — the sequence of chords, when each one lands, and how they line up against the song's beat and bar structure.

> **Status: early development.** The scaffolding, tooling, and conventions are in place; the transcription pipeline is being built out. Interfaces will change without warning.

## Requirements

- Python (version pinned in `pyproject.toml`)
- [uv](https://docs.astral.sh/uv/) for dependency management
- [FFmpeg](https://ffmpeg.org/), available on `PATH`, for decoding audio

## Quickstart

```shell
git clone <repo-url>
cd mangomusic
uv sync
```

That creates the virtual environment and installs both runtime and dev dependencies. Run anything in the project with `uv run`:

## Audio ingestion

MangoMusic can currently decode an FFmpeg-supported audio file into a mono
`float32` NumPy signal at a requested sample rate:

```python
from pathlib import Path

from mangomusic.audio import load_audio

samples, sample_rate_hz = load_audio(Path("song.mp3"), 22_050)
```

Use `start_time_seconds` and `stop_time_seconds` to decode a specific interval.

Chart generation is still under development.

## Rhythm analysis

MangoMusic can compute an onset-strength representation and analyze decoded
audio for a global tempo, ordered beat timestamps, and confidence estimates:

```python
from pathlib import Path

from mangomusic.audio import load_audio
from mangomusic.rhythm import analyze_rhythm, compute_onset_strength

samples, sample_rate_hz = load_audio(Path("song.mp3"), 22_050)
onset_strength, onset_times_seconds = compute_onset_strength(
    samples,
    sample_rate_hz,
)
rhythm = analyze_rhythm(samples, sample_rate_hz)
```

Beat grouping assumes 4/4 time. The first detected beat is labeled beat one of
bar one; this initial implementation does not infer the musical downbeat.
Silent and non-rhythmic inputs return no BPM or beats and zero tempo confidence.

## Chroma features

MangoMusic can fold audio into per-frame pitch-class energy and collapse those
frames onto the beat grid, giving one 12-dimensional chroma vector per beat —
the input a chord recognizer scores against chord templates:

```python
from pathlib import Path

import numpy as np

from mangomusic.audio import load_audio
from mangomusic.chroma import aggregate_chroma_by_beat, compute_chroma
from mangomusic.rhythm import analyze_rhythm

samples, sample_rate_hz = load_audio(Path("song.mp3"), 22_050)
rhythm = analyze_rhythm(samples, sample_rate_hz)
chroma, frame_times_seconds = compute_chroma(samples, sample_rate_hz)
beat_times_seconds = np.array(
    [beat.timestamp_seconds for beat in rhythm.beats],
    dtype=np.float64,
)
beat_chroma = aggregate_chroma_by_beat(
    chroma,
    frame_times_seconds,
    beat_times_seconds,
)
```

Row `0` of a chroma matrix is pitch class C through row `11` for B, matching
`PITCH_CLASS_NAMES`. Pitch is referenced to A440 equal temperament; the tuning
of the recording is not estimated. Frame `index` is centered at
`index * hop_length_samples / sample_rate_hz` seconds, so chroma frames and beat
timestamps share one timeline.

Every chroma vector is L2-normalized, so a dot product against an L2-normalized
chord template is that template's cosine similarity. Vectors with no energy stay
all zeros rather than taking an arbitrary direction, and so score zero against
every template.

Aggregation returns one vector per beat: beat `index` spans up to the next beat,
and the last beat spans through the final feature frame. Frames within a span are
combined with the median by default, which ignores a minority of outlying frames
such as the broadband energy of a note attack; pass
`aggregation=ChromaAggregation.MEAN` for the plain mean. A span holding no frame,
possible when beats fall closer together than the hop length, falls back to the
frame nearest the span midpoint. If rhythm analysis detects no beats, as with
silent or non-rhythmic audio, aggregation returns an empty matrix with shape
`(12, 0)`.

## Chord recognition

MangoMusic can label each beat with a chord by scoring its chroma vector against
a bank of chord templates:

```python
from mangomusic.chords import recognize_chords

analysis = recognize_chords(beat_chroma, beat_times_seconds)
for label in analysis.labels:
    print(f"{label.timestamp_seconds:6.2f}  {label.symbol}  {label.confidence:.2f}")
```

The vocabulary is the 24 major and minor triads plus an explicit no-chord label,
`N`. Each template carries equal weight on its root, third, and fifth and is
L2-normalized, so scoring a chroma vector against it yields a cosine similarity in
`[0.0, 1.0]` — reported as the label's `confidence`. Every template is a rotation
of the C template of the same quality, so recognition is exactly
transposition-equivariant.

A beat is labeled `N` when its best score does not exceed `min_confidence`. The
default of `0.5` is derived rather than tuned: it is the score a flat chroma
vector — one carrying no harmonic information at all — earns against every triad
template, so a beat must beat a uniform spectrum to be given a chord. Silence
scores `0.0` and falls out as `N`. The confidence is reported either way, so a
rejected beat still shows how close it came. Ties resolve to the lowest template
row, keeping output deterministic.

Each beat is labeled independently: there is no smoothing or continuity across
beats, so an ambiguous beat can interrupt a run of one chord. Sevenths, extended
and altered chords, inversions, and bass notes are out of scope for now, as is
key estimation.

## Project layout

```
mangomusic/
├── src/mangomusic/       # package source
│   ├── audio.py          # decoding, resampling, and canonicalization
│   ├── rhythm.py         # onset, tempo, and beat analysis
│   ├── chroma.py         # pitch-class features and beat-synchronous aggregation
│   ├── chords.py         # chord templates and per-beat chord recognition
│   └── errors.py         # public exception hierarchy
├── tests/                # pytest suite
├── docs/
├── AGENTS.md             # instructions for coding agents working in this repo
├── pyproject.toml
└── README.md
```

## Development

Four quality gates, all runnable through `uv`:

```shell
uv run ruff format .      # formatting
uv run ruff check .       # linting
uv run pyright            # type checking
uv run pytest             # tests
```

All four must pass before a change is considered done. `AGENTS.md` has the full definition-of-done gate, git workflow, and prohibited patterns — read it before contributing, and point any coding agent at it too.

## Roadmap

- [x] Audio loading and resampling
- [x] Onset strength, global tempo, and beat tracking
- [ ] Downbeat inference
- [x] Chroma feature extraction and beat-synchronous aggregation
- [x] Chord recognition
- [ ] Beat-aligned segmentation
- [ ] Chord sheet rendering

## Contributing

Work happens on branches with pull requests. Run the quality gates before opening one. Details in `AGENTS.md`.
