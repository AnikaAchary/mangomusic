# MangoMusic

This project is an automatic transcription tool that will take audio input and produce a chord chart as output.

## Goals
Processing audio
Identifying chords/melodies
Identifying BPM (beats per minute) and rhythm
Creating chromogram from chords and beats 
Harmonic analysis
Output as chord chart 

# Coding Style

- Ruff does both linting and formatting. There is no black, isort, or flake8 — don't add them.
- pyright runs in strict mode for type-checking. Annotate all public function signatures.
- Docstrings on public functions state what it does and the units of every argument and return value.
- Prefer simple and focused implementations. Make the smallest possible change to satisfy the request.
- Don't leave commented-out code or dead scaffolding behind.
- Follow existing project patterns and style. Use modern python syntax.
- Do not refactor code unless explicitly asked for.
- Prefer pathlib over manual path manipulation.

## Data models (pydantic v2)

Pydantic is for data that crosses a boundary — config, CLI input, transcription output,
anything serialized or produced outside this process. It is **not** a general
replacement for type annotations on internal functions.

Every structured type in this codebase is a pydantic v2 model. Do not use
`@dataclass`, `NamedTuple`, `TypedDict`, `attrs`, or `pydantic.dataclasses.dataclass`. Inherit from `pydantic.BaseModel`.

**Use pydantic for:** config and settings (`pydantic-settings`); note events and any
result written to disk or JSON; anything parsed from a file, CLI flag, env var, or
external model output. Use only pydantic v2

**Do not use pydantic for:**

- Audio buffers or feature matrices. NumPy arrays stay plain `npt.NDArray[...]` in
  function signatures — they are not model fields. Never set
  `arbitrary_types_allowed=True` just to force an array into a model.

# Environment

Python and dependencies are managed by uv. Never invoke `pip`, `python`, or `pytest`
directly — always go through uv. Use python 3.14.

```sh
uv sync              # sync the venv to uv.lock
uv run <cmd>         # run anything inside the venv
uv add <pkg>         # add a runtime dependency
uv add --dev <pkg>   # add a dev dependency
```

Do not hand-edit dependency entries in `pyproject.toml` or touch `uv.lock` — use
`uv add` / `uv remove` so the lockfile stays consistent. `uv.lock` is committed;
include it in any diff that changes dependencies.

# Definition of done

All four must pass before a task is complete:

```sh
uv run ruff format .
uv run ruff check --fix .
uv run pyright
uv run pytest
```

Fix the cause, don't silence the check. No `# type: ignore`, no `# noqa`, no widening a type to `Any`, no deleting or `xfail`ing a failing test, no `--no-verify`. If a suppression is genuinely correct, add a comment explaining why and flag it in your summary.

# Git

- Branch off `main`. Conventional commit subjects: `feat:`, `fix:`, `test:`
- Small commits; don't mix refactors with behavior changes.
- Never commit: audio files, model weights, `.venv/`, notebook outputs.
- Keep changes scoped to 1 task.
- Do not rewrite git history. Do not force push unless explicitly requested. 
- Summarize important implementation decisions in the PR description.

# Testing

- pytest, in `tests/`, mirroring the package layout.
- **Never commit real music files** — copyright and repo size. Generate fixtures instead.
  A check that genuinely needs real audio reads it from a directory named by
  `MANGOMUSIC_SAMPLES_DIR` (default: `Recording_samples` beside the repo) and **skips** when
  it is absent, so `uv run pytest` stays green without it. Ground truth for those recordings
  lives in `tests/data/recordings.json` — chord names and timestamps only, never audio.
- Never assert exact float equality. Use `np.testing.assert_allclose` with an explicit tolerance chosen for the signal, not copied from another test.
- Every bug fix gets a regression test unless explicitly told to ignore.
- Validation is behavior: every constrained field gets a test asserting that
  out-of-range input raises `ValidationError`.
- Test externally visible behavior rather than implementation behavior.
- Keep deterministic tests.
- Do not remove or weaken tests without explicit permission.

# Dependencies

- Avoid adding dependencies unless they are clearly justified.
- Never silently change requirements, supported Python versions, dependencies, or expected behavior.
- Clearly report any requirement changes and the reason for them.

# Documentation

- Use `README.md` for the project overview, current status, user commands, configuration, and links to detailed documentation.
- `docs/` holds the detailed documentation the README links to. Keep every fact in one place: the detail lives in `docs/`, and the README links to it rather than repeating it. `docs/README.md` indexes the folder — add a row there for any new page.
- When a change affects documented behavior, architecture, database structure, commands, configuration, requirements, or development workflow, update every affected document in the same change. These documentation updates are required and must not be deferred. In practice:
  - a pipeline stage's behavior → `docs/pipeline/<stage>.md`
  - how the stages fit together, or a convention they share → `docs/architecture.md`
  - where code, tests, or docs live → `docs/repo-layout.md`
  - a workflow, testing, or style rule → **this file and `docs/contributing.md`**, which restates these rules for humans and defers to this file as authoritative
  - install steps, requirements, status, or the roadmap → `README.md`
- A new pipeline stage gets its own page under `docs/pipeline/`, following the skeleton the existing pages share: what the stage does, public API, conventions and units, limitations.
- Do not document obvious implementation details.
- If a system change does not require a documentation update, state why in the handoff.

# Repo structure

- Put application code under `src/mangomusic/`.
- Put tests under `tests/`.
- Put detailed documentation under `docs/`, one page per pipeline stage in `docs/pipeline/`.
- Put exploratory notebooks under `notebooks/`. They are for looking at data, not for
  assertions — anything that must keep passing belongs in `tests/`. Notebook outputs are
  stripped on commit by nbstripout; run `uv run nbstripout --install --attributes
  .gitattributes` once per clone.

