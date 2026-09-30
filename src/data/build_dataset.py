"""Convert raw trajectories into prefix instances.

One row per (trajectory, prefix ending at step t). The final step of every
trajectory (always a submit_answer, terminal or synthetic) is dropped; mid-run
submit_answer calls are kept as ordinary steps. A trailing step with an empty
observation is then dropped too: it is the unexecuted action at the turn cap, a
termination artifact that only turn-capped runs have. Every row has
len(steps) == t.

Usage: python -m src.data.build_dataset [--raw-dir DIR] [--out-dir DIR]
"""

import argparse
import json
import os
import re
from collections import Counter
from pathlib import Path

SPLITS = ("train", "valid", "test")
STEP_KEYS = ("step", "think", "graph", "observation", "valid_action")
INPUT_KEYS = ("t", "query", "steps")
# Raw length of a run that exhausted the turn budget; no split has a longer one.
TURN_CAP_STEPS = 52
DEFAULT_RAW_DIR = Path(os.environ.get("MAG_DATA", "~/data/mag-trajectories")).expanduser()
DEFAULT_OUT_DIR = Path("data")


def load_split(raw_dir, split):
    return [json.loads(p.read_text()) for p in sorted((raw_dir / split).glob("q_*.json"))]


def is_submit(step):
    return (step["graph"] or "").startswith("submit_answer")


def gold_ids(traj):
    return {str(g) for g in traj["ground_truth_answer_ids"]}


def ids_in_text(text):
    return set(re.findall(r"\d+", text or ""))


def label_hit1(traj):
    ranked = traj["final_answer"]["ranked_node_ids"]
    return bool(ranked) and str(ranked[0]) in gold_ids(traj)


def label_observed_gold(traj):
    gold = gold_ids(traj)
    return any(gold & ids_in_text(s["observation"]) for s in traj["trajectory"])


def hit_turn_cap(traj):
    return len(traj["trajectory"]) == TURN_CAP_STEPS


def trajectory_labels(traj):
    quality = traj["quality"]
    return {
        "label_accepted": bool(quality["accepted"]),
        "label_hit1": label_hit1(traj),
        "label_observed_gold": label_observed_gold(traj),
        "label_looping": hit_turn_cap(traj),
    }


def prefix_steps(trajectory, t):
    """Steps 1..t, keeping only STEP_KEYS."""
    return [{k: s[k] for k in STEP_KEYS} for s in trajectory[:t]]


def is_empty_observation(step):
    return not (step["observation"] or "").strip()


def cutoffs(trajectory):
    """Every t except the final step's, and except a trailing empty-observation step."""
    last = len(trajectory) - 1
    if last > 0 and is_empty_observation(trajectory[last - 1]):
        last -= 1
    return list(range(1, last + 1))


def build_instances(traj):
    labels = trajectory_labels(traj)
    rows = []
    for t in cutoffs(traj["trajectory"]):
        row = {
            "qid": str(traj["qid"]),
            "split": traj["split"],
            "t": t,
            "query": traj["query"],
            "steps": prefix_steps(traj["trajectory"], t),
        }
        row.update(labels)
        rows.append(row)
    return rows


def model_inputs(row):
    """The only view of a row that feature-building code may receive."""
    return {k: row[k] for k in INPUT_KEYS}


def reject_reasons(trajs):
    return Counter(t["quality"].get("reject_reason") for t in trajs)


def write_jsonl(rows, path):
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w") as f:
        for row in rows:
            f.write(json.dumps(row) + "\n")


def positive_rates(rows):
    keys = [k for k in rows[0] if k.startswith("label_")]
    return {k: sum(r[k] for r in rows) / len(rows) for k in keys}


def build_split(raw_dir, out_dir, split):
    trajs = load_split(raw_dir, split)
    for traj in trajs:
        assert all(s["step"] == i + 1 for i, s in enumerate(traj["trajectory"])), traj["qid"]
        assert is_submit(traj["trajectory"][-1]), f"final step not a submit: {traj['qid']}"
        assert len(traj["trajectory"]) <= TURN_CAP_STEPS, f"longer than turn cap: {traj['qid']}"
    rows = [row for traj in trajs for row in build_instances(traj)]
    write_jsonl(rows, out_dir / f"instances_{split}.jsonl")
    return trajs, rows


def report(split, trajs, rows):
    rates = positive_rates(rows)
    rate_str = ", ".join(f"{k.removeprefix('label_')} {100 * v:.1f}%" for k, v in rates.items())
    print(f"{split:<6s} {len(trajs):5d} trajectories, {len(rows):6d} instances")
    print(f"       positive rates: {rate_str}")
    print(f"       reject reasons: {dict(reject_reasons(trajs).most_common())}")
    capped = [t for t in trajs if hit_turn_cap(t)]
    print(f"       turn-capped: {len(capped)}, by reject reason: {dict(reject_reasons(capped).most_common())}")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--raw-dir", type=Path, default=DEFAULT_RAW_DIR)
    parser.add_argument("--out-dir", type=Path, default=DEFAULT_OUT_DIR)
    args = parser.parse_args()
    for split in SPLITS:
        trajs, rows = build_split(args.raw_dir, args.out_dir, split)
        report(split, trajs, rows)


if __name__ == "__main__":
    main()
