# Chroma features

`mangomusic.chroma` folds audio into pitch-class energy — how much of each of
the twelve pitch classes is sounding, with octave discarded — and then collapses
those frames onto the beat grid. The result is one 12-dimensional vector per
beat: the input a chord recognizer scores against chord templates.

## Public API

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

- `compute_chroma(samples, sample_rate_hz, *, hop_length_samples=512)` →
  per-frame chroma and frame timestamps.
- `aggregate_chroma_by_beat(chroma, frame_times_seconds, beat_times_seconds, *,
  aggregation=ChromaAggregation.MEDIAN)` → one chroma vector per beat.
- `normalize_chroma(chroma)` → the same matrix with unit-length columns. Both
  functions above apply it to their own output; call it directly only when
  building chroma some other way.

## Conventions and units

- **Row order** — row `0` is pitch class C, row `1` is C#, through row `11` for
  B, matching `PITCH_CLASS_NAMES`. `PITCH_CLASS_COUNT` is `12`.
- **Shapes** — per-frame chroma is `(12, frame_count)`; beat-synchronous chroma
  is `(12, beat_count)`. Pitch class is always the first axis and time the
  second.
- **Frame timing** — frame `index` is centered at
  `index * hop_length_samples / sample_rate_hz` seconds. Timestamps are returned
  alongside the matrix, so chroma frames and beat timestamps share one timeline
  and can be compared directly.
- **Hop length** — samples, not seconds. Defaults to
  `DEFAULT_HOP_LENGTH_SAMPLES` (`512`). A smaller hop resolves chord changes
  more sharply at proportionally more compute.
- **Tuning** — pitch is referenced to A440 equal temperament. The tuning of the
  recording is not estimated, so a recording pitched materially away from A440
  smears energy across neighboring pitch classes.
- **Magnitude** — unitless. Energy is summed across octaves by a constant-Q
  transform, so a chord voiced high and the same chord voiced low produce the
  same vector.

## Normalization

Every chroma vector is L2-normalized. A dot product between two unit-length
vectors is their cosine similarity, so scoring a normalized chroma vector
against a normalized chord template yields a similarity directly, and a loud
passage does not outscore a quiet one carrying the same harmony.

Vectors with no energy stay all zeros rather than being given an arbitrary
direction. Such a vector scores `0.0` against every template, which is what
makes silence fall out as no-chord downstream instead of matching whichever
chord an invented direction happened to point at.

## Beat-synchronous aggregation

Beat `index` spans from its own timestamp up to the next beat's; the last beat
spans from its timestamp through the final feature frame. The result therefore
always holds exactly one vector per beat.

Frames within a span are combined with the **median** by default. The median
ignores a minority of outlying frames — most usefully the broadband energy of a
note attack, which carries little harmonic information and would pull a mean
toward every pitch class at once. Pass `aggregation=ChromaAggregation.MEAN` for
the plain mean when you want every frame weighted equally.

Two edge cases have defined behavior:

- **A span holding no frame** — possible when beats fall closer together than
  the hop length — falls back to the single frame nearest the span midpoint, so
  every beat still carries real harmonic content rather than zeros.
- **An empty beat grid**, as returned by rhythm analysis for silent or
  non-rhythmic audio, produces an empty matrix with shape `(12, 0)`.

A column comes back all zeros when its span carries no energy, and, under the
median, when no pitch class sounds in a majority of the span's frames.

## Input requirements

`ValueError` is raised for input that cannot be interpreted, rather than being
silently coerced:

- `chroma` must be two-dimensional with 12 rows and **at least one frame**.
- `frame_times_seconds` must be finite, strictly increasing, and hold one
  timestamp per chroma column.
- `beat_times_seconds` must be finite, nonnegative, and strictly increasing. It
  may be empty — that is the one empty input accepted here.
- All values must be finite.

## Limitations

- No tuning estimation — see above.
- Percussive and broadband content contributes energy across pitch classes; the
  median aggregation mitigates transients but there is no harmonic/percussive
  separation.

## Next stage

[Chord recognition](chords.md) scores each beat's vector against a bank of
chord templates.
