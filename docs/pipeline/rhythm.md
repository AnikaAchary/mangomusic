# Rhythm analysis

`mangomusic.rhythm` finds the pulse: a global tempo, the timestamps of
individual beats, and their positions in an inferred 4/4 bar. Those timestamps
become the grid every later stage is measured against — chroma is aggregated
per beat, and chords are labeled per beat.

## Public API

```python
from pathlib import Path

from mangomusic.audio import load_audio
from mangomusic.rhythm import analyze_rhythm, compute_onset_strength

samples, sample_rate_hz = load_audio(Path("song.mp3"), 22_050)

onset_strength, onset_times_seconds = compute_onset_strength(samples, sample_rate_hz)
rhythm = analyze_rhythm(samples, sample_rate_hz)
```

`compute_onset_strength` exposes the intermediate representation — a
spectral-flux curve that peaks where energy rises — for inspection and plotting.
`analyze_rhythm` computes it internally, so a caller who only wants beats does
not need to call both.

## Results

`analyze_rhythm` returns a `RhythmAnalysis`:

| Field | Type | Meaning |
| --- | --- | --- |
| `bpm` | `float \| None` | Global tempo in beats per minute, or `None` when no tempo was found |
| `tempo_confidence` | `float` | How steady the detected pulse is, in `[0.0, 1.0]` |
| `downbeat_confidence` | `float \| None` | Confidence in the inferred 4/4 phase, or `None` when it is ambiguous |
| `beats` | `list[BeatEvent]` | Detected beats in time order |
| `is_silent` | `bool` | Whether the input was judged silent |

Each `BeatEvent` carries `timestamp_seconds` (seconds from the start of the
analyzed signal), `bar_number` (from `1`), `beat_in_bar` (`1` to `4`), and a
per-beat `confidence` in `[0.0, 1.0]`.

Both models are pydantic and validate their own consistency: an analysis with no
`bpm` may not carry beats, tempo confidence, or downbeat confidence; an
analysis with a `bpm` must carry beats; and a silent analysis may not carry a
`bpm` at all. These combinations cannot be constructed, so callers can branch
on `bpm is None` alone.

## Conventions and units

- **Input** — mono, finite `float32` samples with shape `(sample_count,)` and a
  positive integer sample rate in hertz, exactly as
  [`load_audio`](audio.md) returns them.
- **Onset strength** — unitless values with shape `(onset_frame_count,)`,
  paired with timestamps in seconds of the same shape. Timestamps are returned
  explicitly rather than derived from a frame index, so callers never need to
  know the internal hop length.
- **Beat times** — seconds, strictly increasing. Seconds are the interchange
  unit between stages; frame indices are never passed between modules, because
  each stage frames the signal at its own hop length.

## Bars and downbeats

Beat grouping assumes 4/4 time. Rhythm analysis measures low-, mid-, and
high-frequency spectral-flux accents around every tracked beat, scores the four
possible bar phases, and labels the strongest sufficiently distinct phase as
beat one. It needs at least twelve tracked beats so every phase has evidence
from three bars.

`downbeat_confidence` is a unitless score in `[0.0, 1.0]` based on the winning
phase's separation from the runner-up. When the recording is too short or no
phase is sufficiently distinct, it is `None`. In that ambiguous case, beat and
bar numbering retains the fallback convention that the first tracked beat is
beat one of bar one.

A recording that begins mid-bar or with a pickup can start with a partial bar.
For example, the result may begin with beats three and four of bar one, followed
by beat one of bar two. Bar numbers always begin at one for the analyzed signal.

## Degenerate input

Two cases return no tempo and no beats, distinguished by `is_silent`:

- **Silent** — root-mean-square amplitude at or below the silence threshold.
  Returns `is_silent=True` without running beat tracking.
- **Non-rhythmic** — audio with energy but no trackable pulse, such as a steady
  tone or noise, where beat tracking finds fewer than two beats. Returns
  `is_silent=False`.

Neither raises. `ValueError` is reserved for genuinely unusable input: a
non-positive sample rate, a non-mono or empty array, or non-finite samples.

## Limitations

- One global tempo per input. Tempo changes within a recording are not tracked;
  analyze sections separately using the time range arguments of
  [`load_audio`](audio.md).
- 4/4 only. Other meters are not detected.
- Downbeat inference is an accent-based heuristic. Music without recurring
  downbeat accents, or with heavy syncopation or missing tracked beats, can be
  ambiguous or incorrectly aligned. Check `downbeat_confidence` before relying
  on bar positions.

## Next stage

[Chroma features](chroma.md) uses the beat timestamps to collapse per-frame
pitch-class energy onto this grid.
