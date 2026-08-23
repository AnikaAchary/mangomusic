# Contributing

> **[`AGENTS.md`](../AGENTS.md) is the authoritative version of these rules.**
> This page is the readable introduction; where the two disagree, `AGENTS.md`
> wins. If you are pointing a coding agent at this repository, point it there.

## Environment

Python and dependencies are managed by [uv](https://docs.astral.sh/uv/). Never
invoke `pip`, `python`, or `pytest` directly — always go through uv, so
everything runs against the locked environment.

```sh
uv sync              # sync the venv to uv.lock
uv run <cmd>         # run anything inside the venv
uv add <pkg>         # add a runtime dependency
uv add --dev <pkg>   # add a dev dependency
```

Use `uv add` / `uv remove` rather than editing `pyproject.toml` by hand, and
commit the resulting `uv.lock` alongside the change.

[FFmpeg](https://ffmpeg.org/) must be on `PATH` for the audio tests to run.

## Definition of done

All four gates must pass before a change is complete:

```sh
uv run ruff format .
uv run ruff check --fix .
uv run pyright
uv run pytest
```

Ruff does both linting and formatting — there is no black, isort, or flake8.
pyright runs in strict mode.

**Fix the cause, don't silence the check.** No `# type: ignore`, no `# noqa`, no
widening a type to `Any`, no deleting or `xfail`ing a failing test, no
`--no-verify`. If a suppression is genuinely correct, comment why it is correct
and call it out in the pull request.

## Code style

- Annotate all public function signatures.
- Docstrings on public functions state what the function does and **the units of
  every argument and return value** — hertz for sample rates, samples for hop
  lengths, seconds for timestamps, and the shape of every array.
- Prefer the smallest change that satisfies the request. Don't refactor unless
  that is the task, and don't mix a refactor into a behavior change.
- Leave no commented-out code or dead scaffolding behind.
- Prefer `pathlib` over manual path manipulation.

### Data models

Every structured type is a pydantic v2 model inheriting from `BaseModel`. Do not
use `@dataclass`, `NamedTuple`, `TypedDict`, `attrs`, or
`pydantic.dataclasses.dataclass`.

Pydantic is for data that crosses a boundary — config, CLI input, transcription
output, anything serialized or produced outside this process. It is not a
general replacement for type annotations on internal functions.

Audio buffers and feature matrices are the explicit exception: they stay plain
`npt.NDArray[...]` in function signatures and are never model fields. Never set
`arbitrary_types_allowed=True` to force an array into a model.

## Testing

Tests live in `tests/`, mirroring the package layout.

- **Never commit real music files** — copyright, and repository size. Generate
  fixtures instead: synthesized tones, triads, clicks, and noise.
- **Never assert exact float equality.** Use `np.testing.assert_allclose` with a
  tolerance chosen for the signal at hand, not copied from another test.
- **Validation is behavior.** Every constrained field gets a test asserting that
  out-of-range input raises `ValidationError`.
- **Every bug fix gets a regression test.**
- Test externally visible behavior, not implementation details.
- Keep tests deterministic — seed any randomness.

Do not remove or weaken a test without explicit permission.

## Git

- Branch off `main`. Conventional commit subjects: `feat:`, `fix:`, `test:`.
- Small commits, scoped to one task. Don't mix refactors with behavior changes.
- Never commit audio files, model weights, `.venv/`, or notebook outputs.
- Do not rewrite history, and do not force push unless explicitly asked.
- Summarize important implementation decisions in the pull request description.

## Documentation

When a change affects documented behavior, architecture, commands,
configuration, requirements, or the development workflow, update every affected
document **in the same change** — these updates are not deferred to a follow-up.

In practice that means a change to a pipeline stage updates that stage's page in
[`docs/pipeline/`](pipeline/), a change that alters how stages fit together
updates [architecture.md](architecture.md), and a change to the roadmap or the
install steps updates [`README.md`](../README.md). Don't document obvious
implementation details.
