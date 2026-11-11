#!/usr/bin/env python3
"""
evaluation/prepare_human_eval.py — Blind study packager and Google Form CSV analyzer.

Packages rendered Agentic and Baseline MP4 videos into randomized, strictly blinded
per-participant folders (Pair1_VideoA.mp4 .. Pair3_VideoB.mp4) plus a private
blinding_key.csv / blinding_key.json, and unblinds exported Google Form CSV responses.

Usage:
    # Package blinded evaluation folders for all participants with rendered MP4s
    python3 evaluation/prepare_human_eval.py --form-url "https://forms.gle/YOUR_FORM" --zip

    # Package specific participants only
    python3 evaluation/prepare_human_eval.py --users charles max --form-url "https://forms.gle/YOUR_FORM"

    # Unblind and statistically analyze exported Google Form CSV responses
    python3 evaluation/prepare_human_eval.py --analyze-csv responses.csv --key outputs/human_eval_packages/blinding_key.json
"""

from __future__ import annotations

import argparse
import csv
import json
import random
import re
import shutil
import statistics
import sys
import urllib.parse
import zipfile
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

if __package__ is None:
    sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from evaluation.llm_judge import bootstrap_ci, wilcoxon_p  # noqa: E402


HUMAN_METRICS: tuple[tuple[str, str, str], ...] = (
    ("relevance", "Personal Relevance", r"(relevan|pertinenc)"),
    ("creativity", "Creativity", r"(creativ|créativ)"),
    ("engagement", "Engagement", r"(engag)"),
    ("overall_preference", "Overall Preference", r"(overall|préférenc|preference)"),
)

_SLICE_RE = re.compile(r"^(?P<base>.+?)_part(?P<part>\d+)$", re.IGNORECASE)
_SKIP_DIRS = {"judge", "storyboards", "human_eval_packages"}


def _pick_latest_mp4(folder: Path) -> Path | None:
    if not folder.is_dir():
        return None
    mp4s = sorted(folder.glob("*.mp4"))
    if not mp4s:
        return None
    return mp4s[-1]


def _parse_slice_name(slice_name: str) -> tuple[str, int]:
    match = _SLICE_RE.match(slice_name)
    if not match:
        return slice_name, 1
    return match.group("base"), int(match.group("part"))


def discover_participant_pairs(
    outputs_dir: Path,
    users: list[str] | None = None,
) -> dict[str, list[dict[str, Any]]]:
    """
    Scan outputs_dir for completed Agentic + Baseline MP4 pairs and group them
    by base participant ID (e.g. charles_part1..3 -> 'charles').
    """
    root = Path(outputs_dir)
    if not root.is_dir():
        return {}

    filter_set = set(users) if users else None
    grouped: dict[str, list[dict[str, Any]]] = {}

    for child in sorted(root.iterdir()):
        if not child.is_dir() or child.name in _SKIP_DIRS or child.name.startswith("."):
            continue

        ag_mp4 = _pick_latest_mp4(child / "agentic")
        bl_mp4 = _pick_latest_mp4(child / "baseline")
        if not ag_mp4 or not bl_mp4:
            continue

        base_id, part_num = _parse_slice_name(child.name)
        if filter_set and base_id not in filter_set and child.name not in filter_set:
            continue

        grouped.setdefault(base_id, []).append(
            {
                "slice_name": child.name,
                "part_num": part_num,
                "agentic_mp4": ag_mp4,
                "baseline_mp4": bl_mp4,
            }
        )

    for base_id in grouped:
        grouped[base_id].sort(key=lambda item: (item["part_num"], item["slice_name"]))
    return grouped


def assign_blind_pairs(
    slices: list[dict[str, Any]],
    participant_id: str,
    seed: int = 42,
    shuffle_pairs: bool = True,
) -> list[dict[str, Any]]:
    """
    Create deterministic, balanced A/B randomized pair assignments per participant
    so neither system sits in VideoA consistently while preserving full reproducibility.
    """
    rng = random.Random(f"{seed}:{participant_id}")
    ordered = list(slices)
    if shuffle_pairs and len(ordered) > 1:
        rng.shuffle(ordered)

    n = len(ordered)
    orders = ["AB"] * (n // 2) + ["BA"] * (n // 2)
    if n % 2 == 1:
        orders.append(rng.choice(["AB", "BA"]))
    rng.shuffle(orders)

    assigned: list[dict[str, Any]] = []
    for idx, (item, order) in enumerate(zip(ordered, orders), 1):
        if order == "AB":
            va_sys, vb_sys = "agentic", "baseline"
            va_src, vb_src = item["agentic_mp4"], item["baseline_mp4"]
        else:
            va_sys, vb_sys = "baseline", "agentic"
            va_src, vb_src = item["baseline_mp4"], item["agentic_mp4"]

        assigned.append(
            {
                "pair_num": idx,
                "slice_name": item["slice_name"],
                "order": order,
                "video_a_system": va_sys,
                "video_b_system": vb_sys,
                "video_a_source": str(va_src),
                "video_b_source": str(vb_src),
            }
        )
    return assigned


def _format_form_url(form_url: str, participant_id: str, participant_entry: str) -> str:
    if not form_url:
        return "[INSERT GOOGLE FORM LINK HERE]"
    if not participant_entry:
        return form_url
    parsed = urllib.parse.urlparse(form_url)
    params = urllib.parse.parse_qs(parsed.query, keep_blank_values=True)
    params["usp"] = ["pp_url"]
    entry_key = participant_entry if participant_entry.startswith("entry.") else f"entry.{participant_entry}"
    params[entry_key] = [participant_id]
    new_query = urllib.parse.urlencode(params, doseq=True)
    return urllib.parse.urlunparse(parsed._replace(query=new_query))


def _render_folder_instructions(participant_id: str, pairs: list[dict[str, Any]], form_link: str) -> str:
    pair_lines = "\n".join(
        f"- **Pair {p['pair_num']}:** `Pair{p['pair_num']}_VideoA.mp4` vs `Pair{p['pair_num']}_VideoB.mp4`"
        for p in pairs
    )
    return (
        f"# Video Evaluation Package — Participant `{participant_id}`\n\n"
        f"- **Your Participant ID / Identifiant :** `{participant_id}`\n"
        f"- **Grading Form / Formulaire d'évaluation :** {form_link}\n"
        f"- **Estimated time / Durée estimée :** ~15–20 minutes\n\n"
        f"## Files in This Folder / Contenu du dossier\n"
        f"{pair_lines}\n\n"
        f"---\n\n"
        f"## English — Quick Instructions\n"
        f"1. Turn your **sound ON**. These videos are **storyboard prototypes** (stock photos + voiceover + the bottom text showing the intended visual shot), not finished edited Shorts — please judge the **concept, story, personal fit, and entertainment value**.\n"
        f"2. Open the Google Form above and enter your **Participant ID**: `{participant_id}`.\n"
        f"3. Work through one pair at a time (`Pair 1` → `Pair {len(pairs)}`): watch **Video A** once, rate it (`1–5`) on **Personal Relevance**, **Creativity**, **Engagement**, and **Overall Preference**; then watch and rate **Video B**; then pick your **Pairwise Winner** (`Video A` / `Video B` / `No preference`).\n"
        f"4. Click **Submit** when done. Thank you!\n\n"
        f"---\n\n"
        f"## Français — Instructions rapides\n"
        f"1. Activez le **son**. Ces vidéos sont des **maquettes de storyboards** (photos d'illustration + voix off + texte en bas décrivant le plan visuel prévu), et non des Shorts montés — jugez surtout **l'idée, l'histoire, l'adéquation avec vos goûts et l'intérêt**.\n"
        f"2. Ouvrez le Google Form ci-dessus et saisissez votre **Identifiant Participant** : `{participant_id}`.\n"
        f"3. Évaluez une paire à la fois (`Pair 1` → `Pair {len(pairs)}`) : regardez **Video A** une fois et notez-la (`1–5`) sur **Pertinence personnelle**, **Créativité**, **Engagement** et **Préférence globale** ; puis faites pareil pour **Video B** ; puis choisissez votre **Vidéo préférée** (`Video A` / `Video B` / `Égalité`).\n"
        f"4. Cliquez sur **Envoyer** à la fin. Merci !\n"
    )


def package_human_eval(
    outputs_dir: Path,
    out_dir: Path,
    users: list[str] | None = None,
    seed: int = 42,
    shuffle_pairs: bool = True,
    form_url: str = "",
    participant_entry: str = "",
    zip_packages: bool = False,
) -> dict[str, Any]:
    """
    Build randomized, blind participant folders and master blinding keys.
    Only share <out_dir>/<participant>/ (or its .zip) — NEVER share blinding_key.*.
    """
    grouped = discover_participant_pairs(outputs_dir, users=users)
    if not grouped:
        raise RuntimeError(
            f"No complete agentic+baseline MP4 pairs found under {outputs_dir}"
            + (f" for users={users}" if users else "")
        )

    out_root = Path(out_dir)
    out_root.mkdir(parents=True, exist_ok=True)

    manifest_participants: dict[str, Any] = {}
    csv_rows: list[dict[str, Any]] = []

    for participant_id, slices in sorted(grouped.items()):
        pairs = assign_blind_pairs(slices, participant_id, seed=seed, shuffle_pairs=shuffle_pairs)
        p_dir = out_root / participant_id
        p_dir.mkdir(parents=True, exist_ok=True)

        for stale in p_dir.glob("Pair*_Video*.mp4"):
            stale.unlink()

        for p in pairs:
            dest_a = p_dir / f"Pair{p['pair_num']}_VideoA.mp4"
            dest_b = p_dir / f"Pair{p['pair_num']}_VideoB.mp4"
            shutil.copy2(p["video_a_source"], dest_a)
            shutil.copy2(p["video_b_source"], dest_b)

            csv_rows.append(
                {
                    "participant_id": participant_id,
                    "pair_num": p["pair_num"],
                    "slice_name": p["slice_name"],
                    "order": p["order"],
                    "video_a_file": dest_a.name,
                    "video_b_file": dest_b.name,
                    "video_a_system": p["video_a_system"],
                    "video_b_system": p["video_b_system"],
                    "video_a_source": p["video_a_source"],
                    "video_b_source": p["video_b_source"],
                }
            )

        form_link = _format_form_url(form_url, participant_id, participant_entry)
        guide_path = p_dir / "INSTRUCTIONS_EN_FR.md"
        guide_path.write_text(_render_folder_instructions(participant_id, pairs, form_link), encoding="utf-8")

        zip_path_str = ""
        if zip_packages:
            zip_path = out_root / f"{participant_id}.zip"
            with zipfile.ZipFile(zip_path, "w", compression=zipfile.ZIP_STORED) as zf:
                for item in sorted(p_dir.iterdir()):
                    if item.is_file():
                        zf.write(item, arcname=f"{participant_id}/{item.name}")
            zip_path_str = str(zip_path)

        manifest_participants[participant_id] = {
            "num_pairs": len(pairs),
            "folder": str(p_dir),
            "zip": zip_path_str,
            "form_link": form_link,
            "pairs": pairs,
        }

    key_csv = out_root / "blinding_key.csv"
    fieldnames = [
        "participant_id",
        "pair_num",
        "slice_name",
        "order",
        "video_a_file",
        "video_b_file",
        "video_a_system",
        "video_b_system",
        "video_a_source",
        "video_b_source",
    ]
    with key_csv.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(csv_rows)

    key_json = out_root / "blinding_key.json"
    manifest = {
        "created_at": datetime.now(timezone.utc).isoformat(),
        "seed": seed,
        "shuffle_pairs": shuffle_pairs,
        "outputs_dir": str(outputs_dir),
        "out_dir": str(out_root),
        "participants": manifest_participants,
    }
    key_json.write_text(json.dumps(manifest, indent=2, ensure_ascii=False), encoding="utf-8")
    return manifest


def _find_participant_col(headers: list[str]) -> str:
    for col in headers:
        low = col.lower()
        if "participant" in low or "identifiant" in low:
            return col
    raise ValueError(f"Could not detect Participant ID column in CSV headers: {headers}")


def _find_metric_col(headers: list[str], pair_num: int, slot: str, metric_pattern: str) -> str | None:
    pair_re = re.compile(rf"\bpaire?s?\s*{pair_num}\b", re.IGNORECASE)
    slot_re = re.compile(rf"\bvid[eé]os?\s*{slot}\b", re.IGNORECASE)
    m_re = re.compile(metric_pattern, re.IGNORECASE)
    for col in headers:
        if pair_re.search(col) and slot_re.search(col) and m_re.search(col):
            return col
    return None


def _find_winner_col(headers: list[str], pair_num: int) -> str | None:
    pair_re = re.compile(rf"\bpaire?s?\s*{pair_num}\b", re.IGNORECASE)
    win_re = re.compile(r"(rather watch|winner|prefer|préférez|gagnant|meilleure)", re.IGNORECASE)
    slot_score_re = re.compile(r"\bvid[eé]os?\s*[ab]\s*[:\-—]", re.IGNORECASE)
    for col in headers:
        if pair_re.search(col) and win_re.search(col) and not slot_score_re.search(col):
            return col
    return None


def _parse_int_score(raw: str) -> int | None:
    val = (raw or "").strip()
    if not val:
        return None
    m = re.match(r"^(\d+)", val)
    if not m:
        return None
    score = int(m.group(1))
    return score if 1 <= score <= 5 else None


def _parse_winner_slot(raw: str) -> str | None:
    low = (raw or "").strip().lower()
    if not low:
        return None
    if any(tok in low for tok in ("no preference", "tie", "égal", "egal", "aucune", "both")):
        return "tie"
    if low == "a" or re.search(r"\b(?:vid[eé]o|option)\s*a\b", low):
        return "A"
    if low == "b" or re.search(r"\b(?:vid[eé]o|option)\s*b\b", low):
        return "B"
    return None


def analyze_form_responses(
    csv_path: Path,
    key_path: Path,
    out_dir: Path | None = None,
) -> dict[str, Any]:
    """
    Unblind exported Google Form CSV responses using blinding_key.json and compute
    absolute metric comparisons (bootstrapped 95% CI, paired Wilcoxon) and pairwise win rates.
    """
    key_data = json.loads(Path(key_path).read_text(encoding="utf-8"))
    participants_key = key_data.get("participants", {})

    with Path(csv_path).open("r", encoding="utf-8-sig", newline="") as f:
        reader = csv.DictReader(f)
        headers = list(reader.fieldnames or [])
        rows = list(reader)

    if not rows:
        raise RuntimeError(f"No response rows found in {csv_path}")

    p_col = _find_participant_col(headers)
    unblinded_rows: list[dict[str, Any]] = []
    skipped_rows: list[str] = []

    for row in rows:
        pid = (row.get(p_col) or "").strip()
        if pid not in participants_key:
            skipped_rows.append(pid or "<empty>")
            continue

        for p_info in participants_key[pid].get("pairs", []):
            pnum = int(p_info["pair_num"])
            va_sys = p_info["video_a_system"]
            vb_sys = p_info["video_b_system"]

            rec: dict[str, Any] = {
                "participant_id": pid,
                "pair_num": pnum,
                "slice_name": p_info["slice_name"],
                "order": p_info["order"],
            }

            has_any = False
            for m_key, _m_label, m_pat in HUMAN_METRICS:
                col_a = _find_metric_col(headers, pnum, "A", m_pat)
                col_b = _find_metric_col(headers, pnum, "B", m_pat)
                sc_a = _parse_int_score(row.get(col_a, "")) if col_a else None
                sc_b = _parse_int_score(row.get(col_b, "")) if col_b else None
                if sc_a is not None or sc_b is not None:
                    has_any = True
                by_sys = {va_sys: sc_a, vb_sys: sc_b}
                rec[f"agentic_{m_key}"] = by_sys.get("agentic")
                rec[f"baseline_{m_key}"] = by_sys.get("baseline")

            w_col = _find_winner_col(headers, pnum)
            w_slot = _parse_winner_slot(row.get(w_col, "")) if w_col else None
            if w_slot in ("A", "B", "tie"):
                has_any = True
            winner_sys = (
                "tie" if w_slot == "tie"
                else (va_sys if w_slot == "A" else (vb_sys if w_slot == "B" else ""))
            )
            rec["pairwise_winner"] = winner_sys
            if has_any:
                unblinded_rows.append(rec)

    if not unblinded_rows:
        raise RuntimeError(
            "No matching response records decoded. Check Participant IDs and CSV column names."
        )

    evaluated_users = sorted({r["participant_id"] for r in unblinded_rows})
    lines: list[str] = []

    def log(msg: str = "") -> None:
        print(msg)
        lines.append(msg)

    log("=" * 78)
    log(f"HUMAN EVALUATION SUMMARY (participants={len(evaluated_users)}, pairs={len(unblinded_rows)})")
    log("=" * 78)
    log(f"{'metric':<22}{'agentic':>9}{'baseline':>10}{'delta':>8}{'95% CI (boot)':>20}{'wilcoxon p':>12}")

    metric_stats: dict[str, Any] = {}
    for m_key, _m_label, _pat in HUMAN_METRICS:
        per_user_ag: dict[str, list[int]] = {}
        per_user_bl: dict[str, list[int]] = {}
        for r in unblinded_rows:
            ag_v = r.get(f"agentic_{m_key}")
            bl_v = r.get(f"baseline_{m_key}")
            if ag_v is not None and bl_v is not None:
                per_user_ag.setdefault(r["participant_id"], []).append(ag_v)
                per_user_bl.setdefault(r["participant_id"], []).append(bl_v)

        common = sorted(set(per_user_ag) & set(per_user_bl))
        if not common:
            continue
        a_means = [statistics.mean(per_user_ag[u]) for u in common]
        b_means = [statistics.mean(per_user_bl[u]) for u in common]
        diffs = [x - y for x, y in zip(a_means, b_means)]
        lo, hi = bootstrap_ci(diffs) if len(diffs) > 1 else (float("nan"), float("nan"))
        p = wilcoxon_p(a_means, b_means) if len(diffs) > 1 else None
        ci_str = f"[{lo:+.2f}, {hi:+.2f}]" if len(diffs) > 1 else "n/a"
        p_str = f"{p:.4f}" if p is not None else "n/a"
        log(f"{m_key:<22}{statistics.mean(a_means):>9.2f}{statistics.mean(b_means):>10.2f}{statistics.mean(diffs):>+8.2f}{ci_str:>20}{p_str:>12}")
        metric_stats[m_key] = {
            "n_users": len(common),
            "agentic_mean": statistics.mean(a_means),
            "baseline_mean": statistics.mean(b_means),
            "delta": statistics.mean(diffs),
            "ci_lo": lo,
            "ci_hi": hi,
            "wilcoxon_p": p,
        }

    votes = {"agentic": 0, "baseline": 0, "tie": 0}
    per_user_votes: dict[str, dict[str, int]] = {}
    for r in unblinded_rows:
        w = r.get("pairwise_winner")
        if w in votes:
            votes[w] += 1
            uv = per_user_votes.setdefault(r["participant_id"], {"agentic": 0, "baseline": 0, "tie": 0})
            uv[w] += 1

    decided = votes["agentic"] + votes["baseline"]
    win_rate = (votes["agentic"] / decided) if decided > 0 else 0.0

    user_maj = {"agentic": 0, "baseline": 0, "tie": 0}
    for _u, uv in per_user_votes.items():
        if uv["agentic"] > uv["baseline"]:
            user_maj["agentic"] += 1
        elif uv["baseline"] > uv["agentic"]:
            user_maj["baseline"] += 1
        else:
            user_maj["tie"] += 1

    log()
    log("=" * 78)
    log("PAIRWISE PREFERENCE")
    log("=" * 78)
    log(
        f"Pair votes: agentic={votes['agentic']} | baseline={votes['baseline']} | tie={votes['tie']} "
        f"| Agentic win rate (excl. ties)={win_rate:.1%}"
    )
    log(
        f"Per-user majority: agentic={user_maj['agentic']} | baseline={user_maj['baseline']} | tie={user_maj['tie']}"
    )
    if skipped_rows:
        log(f"Skipped unknown Participant IDs: {skipped_rows}")

    dest_dir = Path(out_dir) if out_dir else Path(csv_path).parent
    dest_dir.mkdir(parents=True, exist_ok=True)
    summary_txt = dest_dir / "human_eval_summary.txt"
    summary_txt.write_text("\n".join(lines) + "\n", encoding="utf-8")

    unblinded_csv = dest_dir / "human_eval_unblinded.csv"
    out_fields = list(unblinded_rows[0].keys())
    with unblinded_csv.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=out_fields)
        writer.writeheader()
        writer.writerows(unblinded_rows)

    return {
        "n_users": len(evaluated_users),
        "n_pairs": len(unblinded_rows),
        "metrics": metric_stats,
        "pairwise_votes": votes,
        "win_rate_excl_ties": win_rate,
        "per_user_majority": user_maj,
        "skipped_rows": skipped_rows,
        "summary_txt": str(summary_txt),
        "unblinded_csv": str(unblinded_csv),
    }


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Package blind video pairs for human evaluation or analyze exported Google Form CSV results."
    )
    parser.add_argument("--outputs-dir", type=str, default="outputs", help="Root outputs directory (default: outputs)")
    parser.add_argument(
        "--out-dir",
        type=str,
        default="outputs/human_eval_packages",
        help="Destination directory for blinded participant folders (default: outputs/human_eval_packages)",
    )
    parser.add_argument("--users", nargs="+", default=None, help="Specific participant IDs or slice names to package")
    parser.add_argument("--seed", type=int, default=42, help="Random seed for deterministic blind A/B order")
    parser.add_argument("--no-shuffle-pairs", action="store_true", help="Keep part1/part2/part3 in sequential Pair 1/2/3 order")
    parser.add_argument("--form-url", type=str, default="", help="Google Form responder URL stamped into folder guides")
    parser.add_argument(
        "--participant-entry",
        type=str,
        default="",
        help="Optional Google Form entry ID (e.g. 'entry.123456') to pre-fill Participant ID cleanly",
    )
    parser.add_argument("--zip", action="store_true", help="Also write <participant>.zip per folder")

    # Analysis mode flags
    parser.add_argument("--analyze-csv", type=str, default="", help="Exported Google Form CSV file to unblind & analyze")
    parser.add_argument(
        "--key",
        type=str,
        default="outputs/human_eval_packages/blinding_key.json",
        help="Path to blinding_key.json generated during packaging",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    if args.analyze_csv:
        res = analyze_form_responses(
            csv_path=Path(args.analyze_csv),
            key_path=Path(args.key),
            out_dir=Path(args.analyze_csv).parent,
        )
        print(f"\nSaved {res['summary_txt']} and {res['unblinded_csv']}")
        return

    manifest = package_human_eval(
        outputs_dir=Path(args.outputs_dir),
        out_dir=Path(args.out_dir),
        users=args.users,
        seed=args.seed,
        shuffle_pairs=not args.no_shuffle_pairs,
        form_url=args.form_url,
        participant_entry=args.participant_entry,
        zip_packages=args.zip,
    )
    print(f"Packaged {len(manifest['participants'])} participant(s) into {manifest['out_dir']}:")
    for pid, info in manifest["participants"].items():
        print(f"  - {pid}: {info['num_pairs']} pair(s) -> {info['folder']}" + (f" ({info['zip']})" if info["zip"] else ""))
    print(f"\n[PRIVATE KEY — DO NOT SHARE WITH SUBJECTS] {Path(manifest['out_dir']) / 'blinding_key.csv'}")


if __name__ == "__main__":
    main()
