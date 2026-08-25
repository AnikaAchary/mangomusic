# Architecture

MangoMusic is a linear pipeline. Each stage is a module of plain functions that
takes what the previous stage returned and adds one layer of interpretation:
samples become a pulse, the pulse plus the spectrum becomes harmony, and harmony
becomes chord labels.

```
                          song.mp3
                             │  audio.load_audio
                             ▼
            samples (sample_count,) + sample_rate_hz
                   ┌─────────┴─────────┐
       rhythm.analyze_rhythm    chroma.compute_chroma
                   │                   │
                   ▼                   ▼
        beat times (beat_count,)   chroma (12, frame_count)
                   │               + frame times (frame_count,)
                   └─────────┬─────────┘
                             │  chroma.aggregate_chroma_by_beat
                             ▼
                 beat chroma (12, beat_count)
                             │  chords.recognize_chords
                             ▼
           ChordAnalysis — one ChordLabel per beat
```

The two feature extractions read the samples independently and only meet at
`aggregate_chroma_by_beat`, so chroma can be computed before, after, or in
parallel with rhythm analysis.

## Stage summary

| Stage | Entry point | Consumes | Produces |
| --- | --- | --- | --- |
| [Audio ingestion](pipeline/audio.md) | `load_audio` | A file path | `(samples, sample_rate_hz)` |
| [Rhythm analysis](pipeline/rhythm.md) | `analyze_rhythm` | Samples | `RhythmAnalysis` with beat timestamps |
| [Chroma features](pipeline/chroma.md) | `compute_chroma` | Samples | `(12, frame_count)` + frame times |
| [Beat aggregation](pipeline/chroma.md#beat-synchronous-aggregation) | `aggregate_chroma_by_beat` | Chroma, frame times, beat times | `(12, beat_count)` |
| [Chord recognition](pipeline/chords.md) | `recognize_chords` | Beat chroma, beat times | `ChordAnalysis` |

## Cross-cutting conventions

These hold across every stage and are what let the modules compose without
knowing about each other.

### Seconds are the interchange unit

Stages frame the signal at different hop lengths — rhythm analysis uses its own
internal hop, chroma defaults to 512 samples and takes the hop as an argument.
Frame indices from one stage therefore mean nothing to another, and are never
passed between modules.

Every stage instead returns **timestamps in seconds** alongside its matrix, and
the timelines are constructed to agree: chroma frame `index` is centered at
`index * hop_length_samples / sample_rate_hz` seconds, measured from the start
of the analyzed signal, and beat timestamps are measured from the same origin.
That shared origin is what makes `aggregate_chroma_by_beat` a matter of
searching one sorted array with another.

One consequence worth remembering: if you decode a section with
`start_time_seconds`, every downstream timestamp is relative to the **start of
that section**, not the start of the file. Add the offset back yourself if you
need file-absolute times.

### Time is the second axis

Every feature matrix is `(feature, time)`: chroma is `(12, frame_count)`, the
template bank is `(24, 12)`, and scores are `(24, frame_count)`. Column `n` is
always the `n`th moment in time.

### Normalization makes scores comparable

Chroma vectors and chord templates are both L2-normalized, which reduces
template scoring to a dot product that yields cosine similarity in `[0.0, 1.0]`.
This is why confidences are comparable between a loud chorus and a quiet verse,
and why a vector with no energy — left as zeros rather than given an arbitrary
direction — scores zero against everything.

### Empty and degenerate input

Silent or non-rhythmic audio is an expected input, not an error, and most of the
pipeline degrades quietly:

1. `analyze_rhythm` returns no `bpm` and no beats, flagging silence with
   `is_silent`.
2. `aggregate_chroma_by_beat` accepts the empty beat grid and returns `(12, 0)`.
3. `recognize_chords` **raises** `ValueError` on that empty matrix — it requires
   at least one beat.

The chain does not run end to end on beatless audio. Guard on the beat grid:

```python
rhythm = analyze_rhythm(samples, sample_rate_hz)
if not rhythm.beats:
    ...  # nothing to transcribe
```

Everywhere else, `ValueError` means the *call* was malformed — a bad shape, a
non-positive sample rate, timestamps that are not strictly increasing — while
`MangoMusicError` and its subclass `AudioDecodeError` (in `mangomusic.errors`)
mean an expected external failure, such as an undecodable file.

### Structured results are pydantic; buffers are not

Anything that crosses a boundary — `RhythmAnalysis`, `BeatEvent`,
`ChordAnalysis`, `ChordLabel` — is a pydantic v2 model, validating its own
invariants so an inconsistent result cannot be constructed or deserialized.
Audio buffers and feature matrices stay plain NumPy arrays in function
signatures and are never model fields. See [contributing](contributing.md) for
the rule in full.

## Where the remaining stages fit

Two stages are not built yet; the [roadmap](../README.md#roadmap) is the source
of truth for status.

- **Downbeat inference** slots into rhythm analysis, replacing the current
  assumption that the first detected beat is beat one of bar one. It changes
  `BeatEvent.bar_number` and `beat_in_bar` only — no downstream shape changes.
- **Beat-aligned segmentation** consumes `ChordAnalysis` and merges runs of
  equal labels into timed chord regions, which is also where continuity across
  beats would be enforced.
- **Chord sheet rendering** turns those regions into the final chart, and is the
  first stage whose output is text rather than data.
