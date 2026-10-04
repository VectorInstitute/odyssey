"""Start the clinician demo (or run its self-check).

Usage, on the GPU host from the repository root::

    .venv/bin/python -m apps.clinician_demo \\
        --run-dir ~/runs/full_run_v10 \\
        --data-dir ~/data/mimiciv_3.1_v1/data/held_out \\
        --metadata-dir ~/data/mimiciv_3.1_v1/metadata \\
        --splits ~/data/mimiciv_3.1_v1/metadata/subject_splits.parquet

then, on the laptop, open a tunnel and browse to http://localhost:8765::

    gcloud compute ssh <vm> --zone <zone> --project <project> \\
        --tunnel-through-iap -- -N -L 8765:localhost:8765

``--data-mode open --data-dir ~/data/mimiciv_demo_meds/data`` serves the
open MIMIC-IV demo instead. ``--self-check`` loads everything, measures one
case end to end, prints a JSON report and exits non-zero on failure.
"""

import argparse
import json
import logging
import sys
from pathlib import Path

from apps.clinician_demo.config import DATA_MODES, DemoConfig


logger = logging.getLogger(__name__)


def parse_args(argv: list[str] | None = None) -> tuple[DemoConfig, bool]:
    """Parse the command line into a config and the self-check flag."""
    parser = argparse.ArgumentParser(
        prog="python -m apps.clinician_demo",
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument("--run-dir", type=Path, required=True)
    parser.add_argument(
        "--data-dir", type=Path, required=True, help="MEDS shard directory"
    )
    parser.add_argument(
        "--metadata-dir", type=Path, default=None, help="MEDS metadata/ (codes.parquet)"
    )
    parser.add_argument(
        "--splits", type=Path, default=None, help="the model's subject_splits.parquet"
    )
    parser.add_argument("--data-mode", choices=DATA_MODES, default="credentialed")
    parser.add_argument("--checkpoint", default="checkpoint_best.pt")
    parser.add_argument("--port", type=int, default=8765)
    parser.add_argument("--alert-rate", type=float, default=0.05)
    parser.add_argument("--max-shards", type=int, default=None)
    parser.add_argument("--cache-dir", type=Path, default=None)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--no-warmup", action="store_true")
    parser.add_argument("--self-check", action="store_true")
    args = parser.parse_args(argv)
    try:
        config = DemoConfig(
            run_dir=args.run_dir.expanduser(),
            data_dir=args.data_dir.expanduser(),
            metadata_dir=args.metadata_dir.expanduser() if args.metadata_dir else None,
            splits_path=args.splits.expanduser() if args.splits else None,
            data_mode=args.data_mode,
            checkpoint=args.checkpoint,
            port=args.port,
            alert_rate=args.alert_rate,
            max_shards=args.max_shards,
            cache_dir=args.cache_dir.expanduser() if args.cache_dir else None,
            device=args.device,
            warmup=not args.no_warmup,
        )
    except ValueError as exc:
        parser.error(str(exc))
    return config, args.self_check


def main(argv: list[str] | None = None) -> int:
    """Run the demo server, or the self-check; return the process exit code."""
    logging.basicConfig(
        level=logging.INFO, format="%(asctime)s %(levelname)s %(name)s: %(message)s"
    )
    config, self_check = parse_args(argv)
    # Deferred: loading torch and the model is the slow part, and argument
    # errors should surface before it.
    from apps.clinician_demo.server import make_server  # noqa: PLC0415
    from apps.clinician_demo.service import DemoService  # noqa: PLC0415

    service = DemoService.from_config(config)
    if self_check:
        try:
            print(json.dumps(service.self_check(), indent=2, default=str))
        except Exception:
            logger.exception("[self-check] FAILED")
            return 1
        finally:
            service.shutdown()
        return 0
    if config.warmup:
        service.warm_up()
    server = make_server(service, config.host, config.port)
    logger.info(
        "serving %s (%s mode) on http://%s:%d -- open an SSH tunnel to this port",
        config.run_name,
        config.data_mode,
        config.host,
        config.port,
    )
    try:
        server.serve_forever()
    except KeyboardInterrupt:
        logger.info("stopping")
    finally:
        server.server_close()
        service.shutdown()
    return 0


if __name__ == "__main__":
    sys.exit(main())
