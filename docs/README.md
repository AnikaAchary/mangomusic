# MangoMusic documentation

Detailed documentation for MangoMusic. The [project README](../README.md) has the
overview, install steps, and current status; these pages have the depth.

## Reading order

Start with the architecture page, then read only the stage you are working on.

| Document | What it covers |
| --- | --- |
| [Architecture](architecture.md) | How the stages compose: what each hands to the next, and the conventions that let them line up |
| [Repository layout](repo-layout.md) | Where code, tests, and docs live, and what belongs in each module |
| [Contributing](contributing.md) | Quality gates, testing conventions, and the git workflow |

## Pipeline stages

In pipeline order — each stage consumes what the one above it produces.

| Stage | Module | What it produces |
| --- | --- | --- |
| [Audio ingestion](pipeline/audio.md) | `mangomusic.audio` | Mono `float32` samples at a requested sample rate |
| [Rhythm analysis](pipeline/rhythm.md) | `mangomusic.rhythm` | Onset strength, a global tempo, beat timestamps, and 4/4 downbeats |
| [Chroma features](pipeline/chroma.md) | `mangomusic.chroma` | Pitch-class energy per frame, then per beat |
| [Chord recognition](pipeline/chords.md) | `mangomusic.chords` | A chord label and confidence for each beat |

Beat-aligned segmentation and chord sheet rendering are not built yet; the
[roadmap](../README.md#roadmap) tracks status.
