"""
build_dataset.py — Phases 2 & 3 driver: build the supervised daily table.

PIPELINE
    load master + single-station sensor  (data_loader)
        -> daily single-station target, >=18h local days  (targets.daily_mean_target)
        -> next-day (t+1) labels on consecutive pairs       (targets.build_supervised_target)
        -> leakage-safe daily features                      (features.build_features)
        -> join features to labels on origin day t
        -> save processed/supervised_daily.parquet
        -> write reports/phase2_feature_dictionary.md and reports/phase3_target.md

The saved table is the single input to Phases 4 and 5. It is written under a NEW name in
processed/ (never overwriting the master).
"""

from __future__ import annotations

import json
from datetime import date

from utils import load_config, set_global_seed, get_paths, ensure_dirs, get_logger
from data_loader import load_master, load_target_sensor
from targets import daily_mean_target, build_supervised_target
from features import build_features, FEATURE_GROUPS

log = get_logger("build_dataset")


def main():
    """Run Phases 2 & 3: build and save the supervised daily table, plus the feature
    dictionary and target reports, then update session memory."""
    cfg = load_config()
    seed = set_global_seed(cfg["project"]["random_seed"])
    ensure_dirs(cfg)
    paths = get_paths(cfg)
    log.info("seed=%s", seed)

    # ---- load ----
    master = load_master(cfg)
    hourly = load_target_sensor(cfg)
    log.info("master %s ; sensor hourly obs %d", master.shape, len(hourly))

    # ---- Phase 3: target ----
    daily = daily_mean_target(hourly, cfg)
    sup = build_supervised_target(daily)
    log.info("qualifying days=%d ; supervised pairs=%d", len(daily), len(sup))

    # ---- Phase 2: features ----
    feats = build_features(master, daily, cfg)

    # ---- join features to labels on origin day t ----
    table = sup.join(feats, how="left")          # keep only rows with a valid t+1 label
    # attach the day-t pm25 (persistence reference) for diagnostics if not already a col
    table = table.join(daily[["pm25_daily_mean", "n_valid_hours"]], how="left")

    out_path = paths["processed_dir"] / "supervised_daily.parquet"
    table.to_parquet(out_path)
    log.info("wrote %s shape=%s", out_path, table.shape)

    # core feature columns (the default model's inputs)
    core_groups = set(cfg["features"]["core_groups"])
    core_cols = [c for c, g in FEATURE_GROUPS.items() if g in core_groups]
    # persist the feature->group map for ablations / Phase 5
    (paths["processed_dir"] / "feature_groups.json").write_text(
        json.dumps({"groups": FEATURE_GROUPS, "core_cols": sorted(core_cols)}, indent=1))

    _write_feature_dictionary(cfg, paths, feats, core_cols)
    _write_target_report(cfg, paths, daily, sup, table)
    _update_memory(cfg, paths, daily, sup, table, core_cols)
    log.info("Phases 2 & 3 complete.")


def _write_feature_dictionary(cfg, paths, feats, core_cols):
    """Write reports/phase2_feature_dictionary.md listing every feature + availability."""
    lines = ["# Phase 2 — Feature Dictionary\n",
             f"_Generated {date.today()}._  Target = next-day daily-mean PM2.5 of sensor "
             "24434 (single station).\n",
             "**Leakage rule:** every feature uses information available **through origin "
             "day t** to predict day **t+1**. The only forward-looking inputs are CALENDAR "
             "features of the target day t+1 — deterministic and known in advance, so they "
             "leak nothing.\n",
             f"\nTotal engineered features: **{feats.shape[1]}**. "
             f"Core-model features (groups {cfg['features']['core_groups']}): "
             f"**{len(core_cols)}**.\n",
             "\n| Feature | Group | In core? | Temporal availability |",
             "|---|---|---|---|"]
    core_set = set(core_cols)
    for c in feats.columns:
        g = FEATURE_GROUPS.get(c, "uncategorised")
        in_core = "yes" if c in core_set else "no"
        if g == "calendar_cyclical":
            avail = "target day t+1 (deterministic)"
        elif "roll" in c or g == "pm25_rolling":
            avail = "trailing window ending day t"
        elif g == "pm25_lags":
            avail = "day t and earlier"
        else:
            avail = "day t (most recent observed)"
        lines.append(f"| `{c}` | {g} | {in_core} | {avail} |")
    lines.append("\n**Justification:** no feature reads any value dated after day t except "
                 "deterministic calendar fields of t+1; PM2.5 rolling windows end at day t "
                 "(target day excluded); FIRMS `firms_frp_sum` is 0-filled where no fire was "
                 "detected (config `preprocessing.firms_frp_sum_fill`).")
    (paths["reports_dir"] / "phase2_feature_dictionary.md").write_text("\n".join(lines),
                                                                       encoding="utf-8")


def _write_target_report(cfg, paths, daily, sup, table):
    """Write reports/phase3_target.md documenting the target definition and counts."""
    per_year = sup.assign(y=table["y_next_day"]).groupby(sup.index.year).size()
    lines = ["# Phase 3 — Target Construction\n",
             "**Definition (verbatim):** `pm25_next_day_daily_mean` = the daily-MEAN PM2.5 "
             "of sensor 24434 on day **t+1**, predicted from data available through day **t**. "
             "Daily mean is over the LOCAL calendar day (Asia/Dhaka, UTC+6); a day counts only "
             "with **≥18 valid hourly observations**. Not an AQI, not a same-hour +24h shift.\n",
             f"- Single-station source: `{cfg['target']['sensor_file']}` "
             f"(sensor {cfg['target']['sensor_id']}, {cfg['target']['station_label']}).\n",
             f"- Qualifying days (≥18 valid hrs): **{len(daily)}**",
             f"- Consecutive (t, t+1) supervised pairs: **{len(sup)}**",
             f"- Origin-day span: **{sup.index.min().date()} → {sup.index.max().date()}**",
             f"- Target mean/std: {table['y_next_day'].mean():.1f} / "
             f"{table['y_next_day'].std():.1f} µg/m³\n",
             "\n## Supervised pairs per origin-day year\n",
             "| Year | Pairs |", "|---|---|"]
    for y, n in per_year.items():
        lines.append(f"| {y} | {n} |")
    lines.append(f"| **Total** | **{len(sup)}** |")
    lines.append(f"\nSaved table: `processed/supervised_daily.parquet` (shape {table.shape}).")
    (paths["reports_dir"] / "phase3_target.md").write_text("\n".join(lines), encoding="utf-8")


def _update_memory(cfg, paths, daily, sup, table, core_cols):
    """Update .claude_memory after Phases 2 & 3 (decisions log + machine state)."""
    mem = paths["memory_dir"]
    dec = mem / "architecture_decisions.md"
    entry = (f"\n## {date.today()} — Phases 2 & 3\n"
             f"- Target: next-day daily-mean PM2.5, single station (sensor 24434), "
             f"≥18h local days. Qualifying days={len(daily)}, supervised pairs={len(sup)}.\n"
             f"- Features: {table.shape[1]-3} engineered; core={len(core_cols)}. "
             f"Leakage rule enforced (features through day t; calendar of t+1 only).\n"
             f"- FIRMS frp_sum 0-filled where no fire. ACAG kept as non-core anchor.\n"
             f"- Saved processed/supervised_daily.parquet.\n")
    with open(dec, "a", encoding="utf-8") as fh:
        fh.write(entry)
    state = {"phase": 3, "status": "complete",
             "last_artifact": "processed/supervised_daily.parquet",
             "supervised_pairs": int(len(sup)), "qualifying_days": int(len(daily)),
             "n_core_features": int(len(core_cols)),
             "open_issues": ["N≈1746 small-sample (accepted)",
                             "holdout anchored to 2025-03-24 last_available"]}
    (mem / "state_tracker.json").write_text(json.dumps(state, indent=2))


if __name__ == "__main__":
    main()
