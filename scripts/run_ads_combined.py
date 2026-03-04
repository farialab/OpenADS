#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""ADS Combined DWI + PWI Pipeline Runner.

This runner orchestrates the two already-verified entrypoints:
  - scripts/run_ads_dwi.py
  - scripts/run_ads_pwi.py

It runs DWI first, optionally copies the DWI brain mask into the PWI preprocess folder,
then runs PWI. This keeps file naming and output conventions consistent with the
verified scripts, which avoids downstream reporting path mismatches.
"""

from __future__ import annotations

import argparse
import logging
import os
import subprocess
import sys
from pathlib import Path
import shutil

# Project setup
PROJECT_ROOT = Path(__file__).resolve().parents[1]
SCRIPTS_DIR = PROJECT_ROOT / "scripts"
sys.path.extend([str(PROJECT_ROOT), str(PROJECT_ROOT / "src")])

from ads.core.config import load_config


def _setup_logging() -> None:
    logging.basicConfig(level=logging.INFO, format="[%(asctime)s] %(levelname)s: %(message)s")


def _run(cmd: list[str], log: logging.Logger, env: dict[str, str] | None = None) -> None:
    log.info("Running command:")
    log.info("  " + " ".join(cmd))
    subprocess.run(cmd, check=True, env=env)


def _maybe_copy_dwi_mask_to_pwi(out_root: Path, subject_id: str, log: logging.Logger) -> None:
    dwi_pp = out_root / subject_id / "DWI" / "preprocess"
    pwi_pp = out_root / subject_id / "PWI" / "preprocess"

    if not dwi_pp.exists():
        log.warning(f"DWI preprocess folder not found, skip mask copy: {dwi_pp}")
        return

    dwi_mask = dwi_pp / f"{subject_id}_DWIbrain-mask.nii.gz"
    if not dwi_mask.exists():
        cand = sorted(dwi_pp.glob(f"{subject_id}*_mask*.nii*"))[:10]
        log.warning(f"DWI mask not found at expected path: {dwi_mask}")
        if cand:
            log.warning("Nearby mask-like files:")
            for p in cand:
                log.warning(f"  - {p}")
        return

    pwi_pp.mkdir(parents=True, exist_ok=True)
    pwi_mask = pwi_pp / dwi_mask.name

    if pwi_mask.exists():
        log.info("PWI already has a DWIbrain-mask, skip copy")
        return

    shutil.copy(dwi_mask, pwi_mask)
    log.info(f"✓ Copied DWI mask to PWI preprocess: {pwi_mask}")


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Run the verified DWI and PWI pipelines as a combined workflow."
    )
    parser.add_argument("--dwi-subject-path", type=Path, required=True, help="Path to DWI subject directory")
    parser.add_argument("--pwi-subject-path", type=Path, required=True, help="Path to PWI subject directory")
    parser.add_argument("--dwi-config", type=Path, default=PROJECT_ROOT / "configs/dwi_pipeline.yaml", help="DWI config YAML")
    parser.add_argument("--pwi-config", type=Path, default=PROJECT_ROOT / "configs/pwi_pipeline.yaml", help="PWI config YAML")
    parser.add_argument("--gpu", type=str, default="0", help="GPU device id (CUDA_VISIBLE_DEVICES)")
    parser.add_argument("--all", action="store_true", help="Run all stages for both pipelines")
    parser.add_argument("--dwi-stages", type=str, help="Comma-separated DWI stages to run (overrides YAML)")
    parser.add_argument("--pwi-stages", type=str, help="Comma-separated PWI stages to run (overrides YAML)")
    parser.add_argument(
        "--output-root",
        type=Path,
        help="Override output root for both DWI and PWI pipelines (overrides config paths.output_root)",
    )
    parser.add_argument("--skip-dwi", action="store_true", help="Skip DWI pipeline run")
    parser.add_argument("--skip-pwi", action="store_true", help="Skip PWI pipeline run")
    parser.add_argument("--no-mask-copy", action="store_true", help="Do not copy DWI brain mask into PWI preprocess folder")

    args = parser.parse_args()
    _setup_logging()
    log = logging.getLogger("ADS.Combined")

    env = os.environ.copy()
    env["CUDA_VISIBLE_DEVICES"] = str(args.gpu)

    if not args.dwi_subject_path.exists():
        raise FileNotFoundError(f"DWI subject path not found: {args.dwi_subject_path}")
    if not args.pwi_subject_path.exists():
        raise FileNotFoundError(f"PWI subject path not found: {args.pwi_subject_path}")
    if not args.dwi_config.exists():
        raise FileNotFoundError(f"DWI config not found: {args.dwi_config}")
    if not args.pwi_config.exists():
        raise FileNotFoundError(f"PWI config not found: {args.pwi_config}")

    subject_id = args.dwi_subject_path.name
    if args.pwi_subject_path.name != subject_id:
        log.warning(f"DWI subject id ({subject_id}) != PWI subject id ({args.pwi_subject_path.name}). Proceeding with DWI id.")

    dwi_cfg = load_config(args.dwi_config)
    pwi_cfg = load_config(args.pwi_config)
    dwi_out_root = Path(dwi_cfg["paths"]["output_root"])
    pwi_out_root = Path(pwi_cfg["paths"]["output_root"])
    if args.output_root is not None:
        dwi_out_root = args.output_root
        pwi_out_root = args.output_root

    if dwi_out_root != pwi_out_root:
        log.warning("DWI and PWI output_root differ. Mask copy will use DWI output_root.")
        log.warning(f"DWI output_root: {dwi_out_root}")
        log.warning(f"PWI output_root: {pwi_out_root}")

    dwi_cmd = [
        sys.executable,
        str(SCRIPTS_DIR / "run_ads_dwi.py"),
        "--config",
        str(args.dwi_config),
        "--subject-path",
        str(args.dwi_subject_path),
        "--gpu",
        str(args.gpu),
    ]
    if args.all:
        dwi_cmd.append("--all")
    elif args.dwi_stages:
        dwi_cmd.extend(["--stages", args.dwi_stages])
    if args.output_root is not None:
        dwi_cmd.extend(["--output-root", str(args.output_root)])

    pwi_cmd = [
        sys.executable,
        str(SCRIPTS_DIR / "run_ads_pwi.py"),
        "--config",
        str(args.pwi_config),
        "--subject-path",
        str(args.pwi_subject_path),
        "--gpu",
        str(args.gpu),
    ]
    if args.all:
        pwi_cmd.append("--all")
    elif args.pwi_stages:
        pwi_cmd.extend(["--stages", args.pwi_stages])
    if args.output_root is not None:
        pwi_cmd.extend(["--output-root", str(args.output_root)])

    run_dwi = not args.skip_dwi and (args.all or bool(args.dwi_stages))
    run_pwi = not args.skip_pwi and (args.all or bool(args.pwi_stages))
    if not run_dwi and not run_pwi:
        raise ValueError("Nothing to run. Use --all or provide at least one of --dwi-stages/--pwi-stages.")

    try:
        log.info("=" * 60)
        log.info("✅ COMBINED PIPELINE START")
        log.info(f"Subject: {subject_id}")
        log.info(f"GPU: {args.gpu}")
        log.info("=" * 60)

        if run_dwi:
            log.info("=" * 60)
            log.info("DWI PIPELINE")
            log.info("=" * 60)
            _run(dwi_cmd, log, env=env)
        else:
            log.info("Skipping DWI pipeline (no DWI stages selected or --skip-dwi).")

        if not args.no_mask_copy and run_dwi:
            _maybe_copy_dwi_mask_to_pwi(dwi_out_root, subject_id, log)

        if run_pwi:
            log.info("=" * 60)
            log.info("PWI PIPELINE")
            log.info("=" * 60)
            _run(pwi_cmd, log, env=env)
        else:
            log.info("Skipping PWI pipeline (no PWI stages selected or --skip-pwi).")

        log.info("=" * 60)
        log.info("✅ COMBINED PIPELINE FINISHED SUCCESSFULLY")
        log.info("=" * 60)

    except subprocess.CalledProcessError as e:
        log.error("=" * 60)
        log.error(f"❌ Combined Pipeline Failed. Command returned code {e.returncode}")
        log.error("=" * 60)
        raise
    except Exception as e:
        log.error("=" * 60)
        log.error(f"❌ Combined Pipeline Failed: {e}")
        log.error("=" * 60)
        raise


if __name__ == "__main__":
    main()
