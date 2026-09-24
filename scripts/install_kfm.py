#!/usr/bin/env python3
"""Install calibrated terminal/residual KFM into PEGASUS-consumable paths."""

from __future__ import annotations

import hashlib
import json
import os
import shutil
import sys
from datetime import datetime, timezone
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]


def sha256_file(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def main() -> None:
    result_root = Path(os.environ.get("KFM_RESULT_ROOT", ROOT / "results" / "kfm_calibration"))
    lkf = Path(os.environ.get("LKF_CHECKPOINT", ROOT / "checkpoints" / "M8.ckpt"))
    terminal_install = Path(
        os.environ.get(
            "TERMINAL_INSTALL",
            ROOT / "results" / "terminal_controlled_koopman_calibration",
        )
    )
    residual_install = Path(
        os.environ.get(
            "RESIDUAL_INSTALL",
            ROOT / "results" / "residual_distribution_koopman_calibration",
        )
    )

    src_term = result_root / "terminal_controlled_koopman_calibration" / "checkpoints" / "calibrated.pt"
    src_res = result_root / "residual_distribution_koopman_calibration" / "checkpoints" / "calibrated.pt"
    for p in (src_term, src_res, lkf):
        if not p.exists():
            raise SystemExit(f"missing required artifact: {p}")

    dst_term_dir = terminal_install / "checkpoints"
    dst_res_dir = residual_install / "checkpoints"
    dst_term_dir.mkdir(parents=True, exist_ok=True)
    dst_res_dir.mkdir(parents=True, exist_ok=True)
    dst_term = dst_term_dir / "calibrated.pt"
    dst_res = dst_res_dir / "calibrated.pt"
    shutil.copy2(src_term, dst_term)
    shutil.copy2(src_res, dst_res)

    for src_dir, dst_parent in (
        (src_term.parent.parent, terminal_install),
        (src_res.parent.parent, residual_install),
    ):
        for name in ("calibration_summary.json", "calibration_geometry.csv"):
            s = src_dir / name
            if s.exists():
                shutil.copy2(s, dst_parent / name)

    record = {
        "frozen_at_utc": datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ"),
        "base_lkf": str(lkf),
        "base_lkf_sha256": sha256_file(lkf),
        "terminal_checkpoint": str(dst_term),
        "terminal_sha256": sha256_file(dst_term),
        "residual_checkpoint": str(dst_res),
        "residual_sha256": sha256_file(dst_res),
        "source_tree": str(result_root),
        "pegasus_env": {
            "LKF_CHECKPOINT": str(lkf),
            "TERMINAL_CHECKPOINT": str(dst_term),
            "RESIDUAL_CHECKPOINT": str(dst_res),
        },
    }
    freeze_path = result_root / "FREEZE_RECORD.json"
    freeze_path.parent.mkdir(parents=True, exist_ok=True)
    freeze_path.write_text(json.dumps(record, indent=2, sort_keys=True) + "\n")

    sys.path.insert(0, str(ROOT))
    from pegasus.residual_distribution_koopman_model import load_residual_distribution_checkpoint

    model, _ = load_residual_distribution_checkpoint(
        dst_res,
        terminal_checkpoint=dst_term,
        base_lkf_checkpoint=lkf,
        device="cpu",
        strict_sha=True,
    )
    print("strict load OK; latent_components=", int(model.base_model.lkf.config.latent_components))
    print("wrote", freeze_path)
    print("installed terminal", dst_term)
    print("installed residual", dst_res)


if __name__ == "__main__":
    main()
