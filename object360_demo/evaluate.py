"""Compare demo CSV output with manually annotated ground_truth.csv."""

from __future__ import annotations

import argparse
import csv
import json
from collections import defaultdict
from pathlib import Path


def read_rows(path: Path):
    with path.open(newline="", encoding="utf-8-sig") as f:
        return list(csv.DictReader(f))


def iou(a, b):
    ax1, ay1, ax2, ay2 = (float(a[k]) for k in ("x1", "y1", "x2", "y2"))
    bx1, by1, bx2, by2 = (float(b[k]) for k in ("x1", "y1", "x2", "y2"))
    overlap = max(0, min(ax2, bx2) - max(ax1, bx1)) * max(0, min(ay2, by2) - max(ay1, by1))
    area_a = max(0, ax2 - ax1) * max(0, ay2 - ay1)
    area_b = max(0, bx2 - bx1) * max(0, by2 - by1)
    return overlap / (area_a + area_b - overlap) if area_a + area_b > overlap else 0.0


def match_rows(ground_truth, predictions, threshold=0.5):
    """Greedy one-to-one matching by view, class, and IoU."""
    candidates = []
    for gi, g in enumerate(ground_truth):
        for pi, p in enumerate(predictions):
            if g["view"] == p["view"] and g["class"].lower() == p["class"].lower():
                score = iou(g, p)
                if score >= threshold:
                    candidates.append((score, gi, pi))
    used_g, used_p, matches = set(), set(), []
    for _, gi, pi in sorted(candidates, reverse=True):
        if gi not in used_g and pi not in used_p:
            used_g.add(gi)
            used_p.add(pi)
            matches.append((ground_truth[gi], predictions[pi]))
    return matches, len(ground_truth) - len(used_g), len(predictions) - len(used_p)


def score(ground_truth, detections, tracks):
    det_matches, missed, extra = match_rows(ground_truth, detections)
    track_matches, _, _ = match_rows(ground_truth, tracks)
    id_history = defaultdict(list)
    for g, p in track_matches:
        id_history[g["object_id"]].append((int(g["view"]), p["track_id"]))
    switches = 0
    reappearances = 0
    for history in id_history.values():
        history.sort()
        for (prev_view, prev_id), (view, track_id) in zip(history, history[1:]):
            switches += track_id != prev_id
            reappearances += view - prev_view > 1
    tp = len(det_matches)
    return {
        "true_positives": tp, "missed_objects": missed, "extra_detections": extra,
        "precision": round(tp / (tp + extra), 4) if tp + extra else None,
        "recall": round(tp / (tp + missed), 4) if tp + missed else None,
        "matched_tracks": len(track_matches), "id_switches_between_matched_views": switches,
        "reappearances_evaluated": reappearances,
        "note": "ID switches count only where the same annotated object is matched in two views. "
                "Missing tracks are reported through matched_tracks, not counted as switches.",
    }


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("run_dir", type=Path)
    parser.add_argument("--ground-truth", type=Path, help="Defaults to run_dir/ground_truth.csv")
    args = parser.parse_args()
    truth_path = args.ground_truth or args.run_dir / "ground_truth.csv"
    truth = read_rows(truth_path)
    results = {}
    for mode in ("online", "batch"):
        results[mode] = score(truth, read_rows(args.run_dir / f"{mode}_detections.csv"),
                              read_rows(args.run_dir / f"{mode}_tracks.csv"))
    out = args.run_dir / "evaluation.json"
    out.write_text(json.dumps(results, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps(results, ensure_ascii=False, indent=2))
    print(f"Saved {out}")


if __name__ == "__main__":
    main()
