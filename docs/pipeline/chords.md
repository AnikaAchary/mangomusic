# Chord recognition

`mangomusic.chords` labels each beat with a chord by scoring its chroma vector
against a bank of chord templates and taking the best match.

## Public API

```python
from mangomusic.chords import recognize_chords

analysis = recognize_chords(beat_chroma, beat_times_seconds)
for label in analysis.labels:
    print(f"{label.timestamp_seconds:6.2f}  {label.symbol}  {label.confidence:.2f}")
```

`beat_chroma` and `beat_times_seconds` come from
[chroma aggregation](chroma.md) and [rhythm analysis](rhythm.md) respectively.

The scoring machinery is public too, for inspection or for building a different
decision rule on top:

- `chord_templates()` → the `(24, 12)` template bank.
- `chord_label_names()` → the 24 chord symbols in template row order.
- `score_chord_templates(chroma)` → `(24, frame_count)` cosine similarities.
  Works on any chroma matrix, not just beat-synchronous ones, and normalizes its
  input first, so un-normalized chroma is scored correctly.

## Results

`recognize_chords` returns a `ChordAnalysis` holding one `ChordLabel` per beat
plus the `min_confidence` that produced them — the threshold travels with the
result, so a serialized analysis records the rule that made it.

| `ChordLabel` field | Type | Meaning |
| --- | --- | --- |
| `beat_index` | `int` | Position in the beat grid, from `0` |
| `timestamp_seconds` | `float` | When the beat lands, in seconds |
| `root` | `str \| None` | Pitch class name, or `None` for no chord |
| `quality` | `ChordQuality \| None` | `MAJOR` or `MINOR`, or `None` for no chord |
| `confidence` | `float` | Best template score, in `[0.0, 1.0]` |

`root` and `quality` are always both set or both absent. The `symbol` property
renders the pair as `"C:maj"` or `"A:min"`, and as `NO_CHORD_SYMBOL` (`"N"`) for
a beat with no chord.

## Vocabulary

The 24 major and minor triads, plus an explicit no-chord label. `CHORD_COUNT` is
`24`; the no-chord label has no template and is assigned by confidence instead.

Template rows `0` through `11` are the major triads rooted at C through B, and
rows `12` through `23` the minor triads in the same root order — the order
`chord_label_names()` returns.

Each template carries equal weight on its root, third, and fifth, zero
elsewhere, and is scaled to unit length. Because chroma vectors are normalized
the same way, a dot product against a template *is* that chord's cosine
similarity, reported as the label's `confidence`. Every template is a rotation of
the C template of the same quality, so recognition is exactly
transposition-equivariant: transposing the input by *n* semitones transposes
every label by *n* semitones and leaves the confidences unchanged.

## The confidence threshold

A beat is labeled `N` when its best score does not exceed `min_confidence`.

The default of `0.5` (`DEFAULT_MIN_CONFIDENCE`) is derived rather than tuned: it
is the score a **flat** chroma vector — one with equal energy in all twelve
pitch classes, carrying no harmonic information at all — earns against every
triad template. A beat must therefore beat a uniform spectrum to be given a
chord. Silence scores `0.0` and falls out as `N` for the same reason.

The comparison uses a small tolerance, so a score sitting exactly on the
threshold within floating-point rounding error counts as no-chord. Without it,
the flat vector that scores precisely `0.5` would be labeled a chord or not
depending on which way the hardware happened to round.

The confidence is reported either way, so a rejected beat still shows how close
it came — useful for deciding whether a different threshold suits your material.

Ties resolve to the lowest template row, keeping output deterministic.

## Input requirements

`ValueError` is raised for:

- `beat_chroma` that is not `(12, beat_count)` with **at least one beat**, or
  that contains non-finite values.
- `beat_times_seconds` that is not one-dimensional, finite, nonnegative, and
  strictly increasing, with exactly one timestamp per chroma column.
- `min_confidence` outside `[0.0, 1.0]`.

Note the empty case: [chroma aggregation](chroma.md) *returns* `(12, 0)` for
audio with no detected beats, but `recognize_chords` *rejects* it. Guard on the
beat grid before recognizing — see the
[architecture notes](../architecture.md#empty-and-degenerate-input).

## Limitations

- **No smoothing.** Each beat is labeled independently, with no continuity
  across beats, so a single ambiguous beat can interrupt a run of one chord.
- **Triads only.** Sevenths, extended and altered chords, and suspensions are
  out of the vocabulary; a beat sounding one is labeled with whichever triad
  scores best.
- **No inversions or bass notes.** Chroma discards octave, so a chord's bass
  note is not represented and inversions are indistinguishable.
- **No key estimation**, and therefore no enharmonic spelling: roots are always
  named with the sharp spellings of `PITCH_CLASS_NAMES`, so D-flat major is
  reported as `C#:maj`.
