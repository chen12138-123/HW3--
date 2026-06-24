from __future__ import annotations

import argparse
from pathlib import Path

try:
    from .final_asset_pipeline import main as build_assets
    from .final_fusion import main as build_fusion
    from .report_builder import build_report
except ImportError:
    from final_asset_pipeline import main as build_assets
    from final_fusion import main as build_fusion
    from report_builder import build_report


def main() -> None:
    parser = argparse.ArgumentParser(description="Run the final HW3 problem 1 asset generation, 3D fusion, and report pipeline.")
    parser.add_argument("--root", type=Path, default=Path(__file__).resolve().parents[1])
    parser.add_argument("--skip-assets", action="store_true", help="Reuse existing output/final_assets results.")
    parser.add_argument("--skip-fusion", action="store_true", help="Reuse existing output/final_fused results.")
    args = parser.parse_args()
    if not args.skip_assets:
        build_assets()
    if not args.skip_fusion:
        build_fusion()
    build_report(args.root)
    print("All outputs and report are ready.")


if __name__ == "__main__":
    main()
