"""Runtime configuration for the clinician demo."""

from dataclasses import dataclass, field
from pathlib import Path
from typing import Literal


DataMode = Literal["credentialed", "open"]
DATA_MODES: tuple[DataMode, ...] = ("credentialed", "open")
LOOPBACK_HOSTS = frozenset({"127.0.0.1", "::1", "localhost"})

#: Alert horizons shown everywhere in the demo (hazard bin edges, so exact).
HORIZONS_HOURS: tuple[float, ...] = (8.0, 24.0, 72.0)


@dataclass(frozen=True)
class DemoConfig:
    """Everything the demo needs to start; built by ``__main__`` from the CLI.

    ``data_mode`` decides what patients are shown and to whom:

    - ``credentialed``: held-out MIMIC-IV patients of the run's own
      extraction. Only for viewers holding PhysioNet MIMIC-IV credentials.
    - ``open``: the openly licensed MIMIC-IV Clinical Database Demo (100
      patients). Safe to show any clinician; most of these patients were
      in the model's training split, which the UI states per patient.
    """

    run_dir: Path
    data_dir: Path
    """Directory of MEDS shards (searched recursively for ``*.parquet``)."""
    metadata_dir: Path | None = None
    """MEDS ``metadata/`` dir holding ``codes.parquet`` (readable labels)."""
    splits_path: Path | None = None
    """The model's own ``subject_splits.parquet``, to flag patients it trained on."""
    data_mode: DataMode = "credentialed"
    checkpoint: str = "checkpoint_best.pt"
    host: str = "127.0.0.1"
    port: int = 8765
    alert_rate: float = 0.05
    """Share of at-risk moments the alert line flags (sets its threshold)."""
    max_shards: int | None = None
    cache_dir: Path | None = None
    """Where derived caches go; defaults to ``<run_dir>/demo_cache``. Holds
    patient-level data, so it must stay on the host."""
    device: str = "cuda"
    warmup: bool = True
    horizons: tuple[float, ...] = field(default=HORIZONS_HOURS)

    def __post_init__(self) -> None:
        """Validate the fields that would otherwise fail late or unsafely."""
        if self.data_mode not in DATA_MODES:
            raise ValueError(
                f"data_mode must be one of {DATA_MODES}, got {self.data_mode!r}"
            )
        if self.host not in LOOPBACK_HOSTS:
            raise ValueError(
                f"host must be a loopback address {sorted(LOOPBACK_HOSTS)}; the demo "
                f"serves patient data and is reached through an SSH tunnel, got {self.host!r}"
            )
        if not 0.0 < self.alert_rate < 1.0:
            raise ValueError(f"alert_rate must be in (0, 1), got {self.alert_rate}")
        if not 0 <= self.port <= 65535:
            raise ValueError(f"port must be in [0, 65535], got {self.port}")
        if self.max_shards is not None and self.max_shards < 1:
            raise ValueError(f"max_shards must be >= 1, got {self.max_shards}")
        if not self.horizons or any(h <= 0 for h in self.horizons):
            raise ValueError(f"horizons must be positive, got {self.horizons}")

    @property
    def resolved_cache_dir(self) -> Path:
        """The cache directory, defaulting under the run directory."""
        return (
            self.cache_dir
            if self.cache_dir is not None
            else self.run_dir / "demo_cache"
        )

    @property
    def run_name(self) -> str:
        """Short run label shown in the UI's provenance line."""
        return self.run_dir.name


__all__ = ["DATA_MODES", "HORIZONS_HOURS", "LOOPBACK_HOSTS", "DataMode", "DemoConfig"]
