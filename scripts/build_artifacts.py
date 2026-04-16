from __future__ import annotations

import argparse
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from backend.app.config import get_settings
from backend.app.services.artifact_builder import ArtifactBuilder


def main() -> None:
    parser = argparse.ArgumentParser(description="Build model, retrieval, and analytics artifacts.")
    parser.add_argument("--force", action="store_true", help="Rebuild all artifacts.")
    parser.add_argument(
        "--skip-if-exists",
        action="store_true",
        help="Exit early when artifacts already exist.",
    )
    args = parser.parse_args()

    settings = get_settings()
    builder = ArtifactBuilder(settings)

    if args.skip_if_exists and builder.artifacts_ready():
        print("Artifacts already exist. Skipping rebuild.")
        return

    builder.ensure(force=args.force)
    print("Artifacts are ready.")


if __name__ == "__main__":
    main()
