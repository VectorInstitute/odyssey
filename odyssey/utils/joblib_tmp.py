"""Keep joblib's memmap scratch off ``/dev/shm``.

joblib memmaps any array over ``max_nbytes`` (1 MB by default) to a
temporary folder so worker processes can share it, and it prefers
``/dev/shm`` when that exists. ``/dev/shm`` is a RAM-backed tmpfs sized at
half of physical memory (42 GB on the A100 boxes), which two concurrent
joblib users can exhaust between them -- and a tmpfs running out raises
``OSError: [Errno 28] No space left on device``, indistinguishable in a
traceback from the disk filling up. That is exactly what killed a 4-hour
MEDS-Tab tabularization on 2026-08-23 (the disk had 277 GB free at the
time, and the "space" came back the instant the process died, because it
was never disk space).

Long-running jobs here call :func:`ensure_joblib_temp_folder` at start-up
so their memmaps land on real disk instead, which is large and shared with
nothing. An operator who has already set ``JOBLIB_TEMP_FOLDER`` keeps it.
"""

import logging
import os
from pathlib import Path
from typing import Optional


logger = logging.getLogger(__name__)

ENV_VAR = "JOBLIB_TEMP_FOLDER"
DEFAULT_DIRNAME = "joblib_tmp"


def ensure_joblib_temp_folder(preferred: Optional[Path] = None) -> Path:
    """Point joblib's memmap scratch at real disk; return the folder in use.

    Respects an existing ``JOBLIB_TEMP_FOLDER`` (an operator's explicit
    choice wins). Otherwise uses ``preferred``, else ``~/joblib_tmp``,
    creating it. Sets the environment variable in this process, so it must
    be called before joblib spawns any worker -- at the top of a script's
    ``main``, not mid-run.
    """
    existing = os.environ.get(ENV_VAR)
    if existing:
        folder = Path(existing)
        folder.mkdir(parents=True, exist_ok=True)
        logger.info("[joblib] using the preset %s=%s", ENV_VAR, folder)
        return folder
    folder = preferred if preferred is not None else Path.home() / DEFAULT_DIRNAME
    folder.mkdir(parents=True, exist_ok=True)
    os.environ[ENV_VAR] = str(folder)
    logger.info(
        "[joblib] %s=%s (keeping memmap scratch off /dev/shm; see this "
        "module's docstring)",
        ENV_VAR,
        folder,
    )
    return folder


__all__ = ["ENV_VAR", "ensure_joblib_temp_folder"]
