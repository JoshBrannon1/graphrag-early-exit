import json
from pathlib import Path
from collections import defaultdict
import statistics


import os
DATA = Path(os.environ.get("MAG_DATA", "/mnt/c/Users/Josh/Downloads/failure-prediction"))
SPLIT = "train"


def load_trajectories(split):
    # Yield one parsed trajectory dict per q_*.json file in the split.
    return [json.loads(f.read_text()) for f in sorted((DATA / split).glob("q_*.json"))]


def accept_rate(trajs):
    """Return (n_total, n_accepted, failure_rate_percent).

    The label lives at traj["quality"]["accepted"] and is a bool.
    "Failure rate" here is just everything that isn't accepted.
    """
    # TODO
    ...
    n_total = len(trajs)
    n_accepted = sum(t["quality"]["accepted"] for t in trajs)
    failure_rate_percent = ((n_total - n_accepted) / n_total)* 100
    return (n_total, n_accepted, failure_rate_percent)
    




def hit_at_1_among_wrong_answer(trajs):
    """Of trajectories with reject_reason == "wrong_answer", what fraction
    rank a gold node first?

        gold   -> traj["ground_truth_answer_ids"]          (unordered SET)
        ranked -> traj["final_answer"]["ranked_node_ids"]  (ordered LIST)

    hit@1 == is ranked[0] a member of the gold set?
    """
    num_wrong_answers_with_gold_node_first = 0
    num_wrong_answers = 0
    for traj in trajs:
        if traj["quality"].get("reject_reason") == "wrong_answer":
            num_wrong_answers += 1
            string_gold = {str(item) for item in traj["ground_truth_answer_ids"]}
            if traj["final_answer"]["ranked_node_ids"]:
                if str(traj["final_answer"]["ranked_node_ids"][0]) in string_gold:
                    num_wrong_answers_with_gold_node_first += 1
    return num_wrong_answers_with_gold_node_first/num_wrong_answers


def steps_by_reject_reason(trajs):
    """For each reject_reason, report:
         - count of trajectories
         - median number of steps          (len(traj["trajectory"]))
         - total steps contributed
         - that total as a % of all steps in the split
    """
    steps_by_reason = defaultdict(list)
    for traj in trajs:
        reason = traj["quality"].get("reject_reason")
        steps_by_reason[reason].append(len(traj["trajectory"]))
    grand_total_steps = sum(sum(v) for v in steps_by_reason.values())
    print(f"    {"reason":<26s} {"n":5} {"median":8.1} {"total":10} {"pct":7.1}")
    for reason, steps in sorted(steps_by_reason.items(), key=lambda kv: sum(kv[1]), reverse=True):
        n = len(steps)
        median = statistics.median(steps)
        total = sum(steps)
        pct = 100 * total/grand_total_steps
        print(f"{reason:<26s} {n:5d} {median:8.1f} {total:10,d} {pct:7.1f}%")

if __name__ == "__main__":
    trajs = list(load_trajectories(SPLIT))
    print(f"loaded {len(trajs)} trajectories from {SPLIT}/\n")

    print("accept_rate:", accept_rate(trajs))
    print("hit@1 among wrong_answer:", hit_at_1_among_wrong_answer(trajs))
    print()
    steps_by_reject_reason(trajs)


# ---------------------------------------------------------------------------
# EXPECTED (train split)
#
#   trajectories loaded ................ 2000
#   accepted ........................... 211  (10.55%)
#   failure rate under `accepted` ...... 89.45%   <- email claim #1
#
#   wrong_answer trajectories .......... 798
#   ...of which rank a gold node 1st ... 46.6%    <- email claim #2
#
#   reject_reason      n     median steps   total steps   % of all steps
#   -----------------------------------------------------------------------
#   wrong_answer      798         7             5,865          13.9%
#   no_submit_answer  582        52            30,264          71.6%   <- #3
#   invalid_graph_ex  404         9             4,626          10.9%
#   valid             211         6             1,465           3.5%
#   invalid_graph_syn   5         8                41           0.1%
#
#   total steps in split ............... 42,261
#
# ---------------------------------------------------------------------------
