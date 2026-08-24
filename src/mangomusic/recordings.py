"""Locating sample recordings and describing the content expected in them.

The sample recordings are not part of the repository: they are real audio, which
AGENTS.md keeps out of git. This module resolves them from a directory named by
the ``MANGOMUSIC_SAMPLES_DIR`` environment variable so that tests and notebooks
agree on which file is which, and parses the hand-written manifest that records
what was played in each one.
"""

import json
import os
from pathlib import Path
from typing import Self

from pydantic import BaseModel, Field, model_validator

from mangomusic.chords import chord_label_names, chord_templates
from mangomusic.chroma import PITCH_CLASS_NAMES

SAMPLES_DIR_ENV_VAR = "MANGOMUSIC_SAMPLES_DIR"
DEFAULT_SAMPLES_DIR_NAME = "Recording_samples"


class RecordingWindow(BaseModel):
    """A span of one recording whose harmonic content is known in advance.

    Exactly one of ``chord`` and ``notes`` names the expected content, so a
    strummed triad and a single sustained note can both be described.
    """

    start_seconds: float = Field(ge=0.0)
    stop_seconds: float = Field(gt=0.0)
    chord: str | None = None
    notes: tuple[str, ...] | None = None
    expect_in_tune: bool = True

    @model_validator(mode="after")
    def _validate_window(self) -> Self:
        if self.stop_seconds <= self.start_seconds:
            raise ValueError("stop_seconds must be greater than start_seconds")
        if (self.chord is None) == (self.notes is None):
            raise ValueError("exactly one of chord and notes must be set")
        if self.chord is not None and self.chord not in chord_label_names():
            raise ValueError("chord must name a recognized chord")
        if self.notes is not None:
            if len(self.notes) == 0:
                raise ValueError("notes must not be empty")
            if any(note not in PITCH_CLASS_NAMES for note in self.notes):
                raise ValueError("notes must name pitch classes")
        return self

    @property
    def expected_pitch_classes(self) -> tuple[int, ...]:
        """The pitch classes that should sound, as indices into ``PITCH_CLASS_NAMES``.

        Returns:
            Ascending pitch-class indices in ``[0, 12)``.
        """
        if self.notes is not None:
            return tuple(sorted(PITCH_CLASS_NAMES.index(note) for note in self.notes))
        assert self.chord is not None
        row = chord_label_names().index(self.chord)
        return tuple(
            int(index) for index in sorted(chord_templates()[row].nonzero()[0])
        )


class RecordingManifest(BaseModel):
    """Ground truth for every sample recording, keyed by file name."""

    recordings: dict[str, list[RecordingWindow]]

    @classmethod
    def from_path(cls, manifest_path: Path) -> RecordingManifest:
        """Parse a manifest from a JSON file.

        Args:
            manifest_path: Path to a JSON file holding a serialized manifest.

        Returns:
            The parsed manifest.

        Raises:
            ValidationError: If the file does not describe a usable manifest.
        """
        return cls.model_validate(json.loads(manifest_path.read_text()))


def samples_dir() -> Path | None:
    """Locate the directory holding the sample recordings.

    Reads ``MANGOMUSIC_SAMPLES_DIR`` when it is set, and otherwise falls back to
    a ``Recording_samples`` directory beside the repository, which is where the
    recordings live during local development.

    Returns:
        The directory, or ``None`` when it does not exist. Callers are expected
        to skip work that needs the recordings rather than fail.
    """
    configured = os.environ.get(SAMPLES_DIR_ENV_VAR)
    if configured:
        candidate = Path(configured).expanduser()
    else:
        repo_root = Path(__file__).resolve().parents[2]
        candidate = repo_root.parent / DEFAULT_SAMPLES_DIR_NAME
    return candidate if candidate.is_dir() else None


def recording_path(file_name: str) -> Path | None:
    """Locate one sample recording by file name.

    Args:
        file_name: Name of the recording within the samples directory.

    Returns:
        The path to the recording, or ``None`` when the samples directory or the
        recording itself is absent.
    """
    directory = samples_dir()
    if directory is None:
        return None
    candidate = directory / file_name
    return candidate if candidate.is_file() else None
