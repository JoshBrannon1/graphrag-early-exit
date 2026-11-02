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
# Boolean label columns, i.e. valid training targets. label_first_gold_step is oracle
# metadata (int or None), not a target.
BINARY_LABELS = (
    "label_accepted",
    "label_hit1",
    "label_ever_finds_gold",
    "label_gold_still_ahead",
    "label_submits_gold",
    "label_hits_cap",
)
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


def first_gold_step(traj):
    """Step number of the first kept step whose observation names a gold id, or None.

    Kept steps are the ones that appear in prefixes (see cutoffs). This is the one
    place first-sighting matching is done.
    """
    gold = gold_ids(traj)
    kept = traj["trajectory"][: max(cutoffs(traj["trajectory"]), default=0)]
    for s in kept:
        if gold & ids_in_text(s["observation"]):
            return s["step"]
    return None


def label_submits_gold(traj):
    submitted = {str(i) for i in traj["final_answer"]["submitted_node_ids"]}
    return bool(submitted & gold_ids(traj))


def hit_turn_cap(traj):
    return len(traj["trajectory"]) == TURN_CAP_STEPS


def trajectory_labels(traj):
    quality = traj["quality"]
    first = first_gold_step(traj)
    return {
        "label_accepted": bool(quality["accepted"]),
        "label_hit1": label_hit1(traj),
        "label_first_gold_step": first,
        "label_ever_finds_gold": first is not None,
        "label_submits_gold": label_submits_gold(traj),
        "label_hits_cap": hit_turn_cap(traj),
    }


def gold_still_ahead(labels, t):
    """True if the run will first sight gold after step t."""
    return labels["label_ever_finds_gold"] and t < labels["label_first_gold_step"]


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
        row["label_gold_still_ahead"] = gold_still_ahead(labels, t)
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
    return {k: sum(r[k] for r in rows) / len(rows) for k in BINARY_LABELS}


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
    mid_submits = sum(is_submit(s) for t in trajs for s in t["trajectory"][:-1])
    print(f"       mid-run submit_answer steps: {mid_submits}")
    runs = {r["qid"]: r for r in rows}.values()
    print(f"       runs: ever_finds_gold {sum(r['label_ever_finds_gold'] for r in runs)}, "
          f"submits_gold {sum(r['label_submits_gold'] for r in runs)}, "
          f"hits_cap {sum(r['label_hits_cap'] for r in runs)}")
    firsts = sorted(r["label_first_gold_step"] for r in runs if r["label_ever_finds_gold"])
    print(f"       first_gold_step (finding runs): median {firsts[len(firsts) // 2]}, "
          f"p90 {firsts[int(0.9 * len(firsts))]}, max {firsts[-1]}")
    ahead = Counter(r["t"] for r in rows if r["label_gold_still_ahead"])
    n_ahead = sum(ahead.values())
    early = sum(n for t, n in ahead.items() if t <= 8)
    print(f"       gold_still_ahead: {n_ahead} instances, t=1..10 {[ahead[t] for t in range(1, 11)]}, "
          f"{100 * early / n_ahead:.1f}% at t<=8")


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
