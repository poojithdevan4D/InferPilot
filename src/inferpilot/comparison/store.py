"""Content-addressed filesystem catalog for baseline-eligible run bundles."""

from __future__ import annotations

import hashlib
import json
import os
import shutil
import tempfile
from dataclasses import dataclass
from pathlib import Path

from ..results import ExperimentResult


@dataclass(frozen=True)
class IngestedRun:
    run_id: str
    path: Path
    created: bool


class ResultStore:
    """Persist immutable run bundles without introducing a database."""

    def __init__(self, root: str | os.PathLike[str]) -> None:
        self.root = Path(root)
        self.objects_dir = self.root / "objects"
        self.objects_dir.mkdir(parents=True, exist_ok=True)

    @staticmethod
    def _result_path(run_dir: Path) -> Path:
        path = run_dir / "result.json"
        if not path.is_file():
            raise FileNotFoundError(f"missing result.json in {run_dir}")
        return path

    @staticmethod
    def _file_digest(path: Path) -> str:
        digest = hashlib.sha256()
        with path.open("rb") as stream:
            for chunk in iter(lambda: stream.read(1024 * 1024), b""):
                digest.update(chunk)
        return digest.hexdigest()

    @classmethod
    def _bundle_manifest(cls, run_dir: Path) -> dict[str, str]:
        manifest: dict[str, str] = {}
        for path in sorted(run_dir.rglob("*")):
            if path.is_file() and path.name != "manifest.json":
                manifest[path.relative_to(run_dir).as_posix()] = cls._file_digest(path)
        return manifest

    @staticmethod
    def _run_id(manifest: dict[str, str]) -> str:
        payload = json.dumps(manifest, sort_keys=True, separators=(",", ":")).encode()
        return hashlib.sha256(payload).hexdigest()

    def ingest(self, run_dir: str | os.PathLike[str]) -> IngestedRun:
        """Validate and atomically copy one eligible run bundle into the store."""
        source = Path(run_dir).resolve()
        result_bytes = self._result_path(source).read_bytes()
        result = ExperimentResult.model_validate_json(result_bytes)
        if not result.is_baseline_eligible:
            raise ValueError(f"run is not baseline-eligible: {source}")

        manifest = self._bundle_manifest(source)
        run_id = self._run_id(manifest)
        target = self.objects_dir / run_id
        if target.exists():
            return IngestedRun(run_id=run_id, path=target, created=False)

        temporary = Path(tempfile.mkdtemp(prefix=".ingest-", dir=self.objects_dir))
        try:
            shutil.copytree(source, temporary, dirs_exist_ok=True)
            (temporary / "manifest.json").write_text(
                json.dumps(manifest, sort_keys=True, indent=2)
            )
            os.replace(temporary, target)
        except Exception:
            shutil.rmtree(temporary, ignore_errors=True)
            raise
        return IngestedRun(run_id=run_id, path=target, created=True)

    def load(self, run_id: str) -> ExperimentResult:
        run_dir = self.objects_dir / run_id
        result_path = self._result_path(run_dir)
        manifest_path = run_dir / "manifest.json"
        if not manifest_path.is_file():
            raise ValueError(f"stored run has no integrity manifest: {run_id}")
        try:
            manifest = json.loads(manifest_path.read_text())
        except (json.JSONDecodeError, OSError) as exc:
            raise ValueError(f"stored run has an invalid manifest: {run_id}") from exc
        if not isinstance(manifest, dict) or self._run_id(manifest) != run_id:
            raise ValueError(f"stored run failed manifest integrity check: {run_id}")
        actual = self._bundle_manifest(run_dir)
        if actual != manifest:
            raise ValueError(f"stored run failed bundle integrity check: {run_id}")
        payload = result_path.read_bytes()
        return ExperimentResult.model_validate_json(payload)

    def list_runs(self, experiment_id: str | None = None) -> list[tuple[str, ExperimentResult]]:
        runs: list[tuple[str, ExperimentResult]] = []
        for path in sorted(self.objects_dir.glob("*/result.json")):
            run_id = path.parent.name
            result = self.load(run_id)
            if experiment_id is None or result.config.experiment_id == experiment_id:
                runs.append((run_id, result))
        return runs
