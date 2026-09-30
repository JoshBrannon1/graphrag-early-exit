"""Replay the trivial policy "exit if still running after C agent steps" on valid.

Agent steps are every step except the final one, which is the terminal submit or
the synthetic one appended during cleaning. A run with more than C agent steps is
truncated: it saves (agent steps - C) and is counted as losing its answer.

Usage: python -m src.eval.cap_policy [--caps 8 10 12 15 20 25 30]
"""

import argparse

from src.data.build_dataset import DEFAULT_RAW_DIR, label_hit1, load_split

SPLIT = "valid"
DEFAULT_CAPS = (8, 10, 12, 15, 20, 25, 30)


def agent_steps(traj):
    return len(traj["trajectory"]) - 1


def run_outcomes(trajs):
    """(agent_steps, accepted, hit1) per run."""
    return [(agent_steps(t), bool(t["quality"]["accepted"]), label_hit1(t)) for t in trajs]


def replay_cap(outcomes, cap):
    truncated = [(n, acc, hit) for n, acc, hit in outcomes if n > cap]
    return {
        "steps_saved": sum(n - cap for n, _, _ in truncated),
        "accepted_lost": sum(acc for _, acc, _ in truncated),
        "hit1_lost": sum(hit for _, _, hit in truncated),
    }


def pct(part, whole):
    return 100 * part / whole if whole else 0.0


def report(outcomes, caps):
    total_steps = sum(n for n, _, _ in outcomes)
    n_acc = sum(acc for _, acc, _ in outcomes)
    n_hit = sum(hit for _, _, hit in outcomes)
    print(f"{SPLIT}: {len(outcomes)} runs, {total_steps} agent steps, "
          f"{n_acc} accepted, {n_hit} hit@1")
    print("Losses are an UPPER BOUND: every truncated run is counted as losing its answer,")
    print("but a capped run would still get a forced submit and could keep it.\n")
    print("C | steps saved | accepted lost | hit@1 lost")
    for cap in caps:
        r = replay_cap(outcomes, cap)
        print(f"{cap} | {pct(r['steps_saved'], total_steps):.1f}% | "
              f"{r['accepted_lost']}/{n_acc} ({pct(r['accepted_lost'], n_acc):.1f}%) | "
              f"{r['hit1_lost']}/{n_hit} ({pct(r['hit1_lost'], n_hit):.1f}%)")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--caps", type=int, nargs="+", default=list(DEFAULT_CAPS))
    args = parser.parse_args()
    report(run_outcomes(load_split(DEFAULT_RAW_DIR, SPLIT)), args.caps)


if __name__ == "__main__":
    main()
