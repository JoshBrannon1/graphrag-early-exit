# Dataset notes — MAG GraphDancer trajectories

Reference for anyone (human or agent) touching `src/data/` or `src/eval/`.
All numbers below were reproduced from the train split by `scripts/verify_stats.py`.

## Layout

```
$MAG_DATA/
├── train/q_{qid}.json      2000 files, 51 MB
├── valid/q_{qid}.json       500 files, 13 MB
├── test/q_{qid}.json        500 files, 14 MB
└── subgraphs/{split}/{qid}/subgraph.json    ~1 MB each, 2.9 GB total
```

Note the subgraph path drops the `q_` prefix: `train/q_9071.json` pairs with
`subgraphs/train/9071/subgraph.json`.

## Trajectory schema

Top level: `qid`, `split`, `query`, `ground_truth_answer_ids`, `trajectory`,
`observed_node_ids`, `observed_target_node_ids`, `final_answer`, `quality`,
`shortest_path`, `prompts`, `model`, `status`.

Each `trajectory[t]`:

| Field | Meaning |
| --- | --- |
| `step` | 1-indexed step number |
| `think` | model reasoning for this step |
| `graph` | the tool call: `RetrieveNode`, `NeighbourCheck`, `NodeFeature`, `NodeDegree`, `submit_answer` |
| `observation` | environment's return message |
| `model_output` | raw `<think>...</think><graph>...</graph>` |
| `valid_action` | 1 / 0 / None |
| `done` | terminal flag |

## Failure modes (train)

| `reject_reason` | n | median steps | total steps | % of all steps | dup-action rate |
| --- | --- | --- | --- | --- | --- |
| `no_submit_answer` | 582 | 52 (always) | 30,264 | 71.6% | 0.515 |
| `wrong_answer` | 798 | 7 | 5,862 | 13.9% | 0.008 |
| `invalid_graph_execution` | 404 | 9 | 4,627 | 10.9% | 0.070 |
| `valid` (success) | 211 | 6 | 1,467 | 3.5% | 0.010 |
| `invalid_graph_syntax` | 5 | 8 | 41 | 0.1% | 0.000 |

Grand total 42,261 steps. Accept rate 211/2000 = 10.55%.

Other splits contain reject reasons train does not (`timeout` in valid,
`missing_think_graph_format` in test). **Never hardcode the category list** — derive it
from the data.

Key asymmetry: `no_submit_answer` is 29.1% of trajectories but **71.6% of the compute**.
Those runs loop — over half their actions are literal repeats of an earlier action.
By contrast `wrong_answer` runs finish in ~7 steps with median observation-level gold
recall of 1.0: they saw every correct node and submitted the wrong subset. Early exit
cannot help those.

## Turn cap

`len(trajectory) == 52` means the run hit the 50-turn cap: 51 agent steps plus the
final submit. No run in any split is longer. The invariant is the cap, not the reject
reason:

| split | turn-capped | of which `no_submit_answer` | other |
| --- | --- | --- | --- |
| train | 582 | 582 | — |
| valid | 151 | 150 | qid 2186 (`timeout`) |
| test | 172 | 171 | qid 1122 (`missing_think_graph_format`) |

The final step's `valid_action is None` iff the run is turn-capped, in every split.

**Step-51 artifact.** Step 51 of a turn-capped run is the action generated at the cap
and never executed, so its observation is empty. That holds for every
`no_submit_answer` run and for test qid 1122. Valid qid 2186 is the exception: its last
steps are malformed outputs, so step 51's observation is a malformed-output error
message instead. No other run has an empty observation anywhere, and no empty
observation occurs mid-run — so an empty observation in a prefix identifies a looper.

## Prefix instances

`src/data/build_dataset.py` writes one instance per (trajectory, prefix ending at step
`t`), with `len(steps) == t`. It drops the final step, then a trailing
empty-observation step (the step-51 artifact). Mid-run `submit_answer` calls (failed,
`valid_action=0`) stay as ordinary steps. There are 8 such steps: 5 train, 3 test (test
qid 10822 has two), all `submit_answer([])`. Every one has the observation `"Invalid
submit_answer arguments. Expected one or more node IDs. ..."`, an error string rather
than the final-submit blob (see [Final submit observation](#final-submit-observation)), so
keeping them is safe. `tests/test_labels.py::test_mid_run_submits_are_error_strings`
guards this, and the builder prints the per-split count.

| split | trajectories | instances |
| --- | --- | --- |
| train | 2000 | 39,679 |
| valid | 500 | 9,960 |
| test | 500 | 10,893 |

`label_hits_cap` (renamed from `label_looping`, definition unchanged) = ran to the turn
cap (raw `len(trajectory) == 52`). It equals `reject_reason == "no_submit_answer"` except
for valid qid 2186 and test qid 1122. Instance-level positive rate: 73.3% train, 75.8%
valid. These counts and the turn-capped breakdown are printed by
`python -m src.data.build_dataset`.

### Final submit observation

The final `submit_answer` step's observation is an environment JSON blob, not an agent
observation: `{"ranking_method": "bm25", "candidate_source": "observed_target_node_ids",
"submitted_node_ids": [...], "observed_node_ids": [...], "observed_target_node_ids":
[...], ...}`. It carries the **whole-run aggregates** (leak 4 below), and for a
turn-capped run the synthetic submit lists every observed node. **Dropping the final
step is load-bearing**, not cosmetic. `tests/test_labels.py::test_no_submit_blob_in_prefix`
asserts that no prefix observation contains `ranking_method`, `candidate_source` or
`observed_target_node_ids`.

## Labels: first gold sighting

Target: will the search observe ≥ 1 gold node? A gold node counts as found when its id
appears in the observation of a **kept** step (one that appears in prefixes). No run has
gold only in a dropped step. Column definitions are in CLAUDE.md §3. Every number below
is printed by `python -m src.data.build_dataset`.

| | train | valid | test |
| --- | --- | --- | --- |
| runs that find gold (`label_ever_finds_gold`) | 1,409 / 2,000 (70.45%) | 359 / 500 (71.8%) | 355 / 500 (71.0%) |
| ... instance rate | 59.0% | 61.9% | 61.1% |
| `label_first_gold_step` median / p90 / max (finding runs) | 2 / 5 / 40 | 2 / 5 / 24 | 2 / 5 / 25 |
| `label_gold_still_ahead` instances | 2,580 (6.5%) | 613 (6.2%) | 640 (5.9%) |
| ... share at t ≤ 8 | 92.1% | 91.4% | 94.1% |
| runs that submit gold (`label_submits_gold`) | 923 (46.15%) | 232 (46.4%) | 224 (44.8%) |

`label_gold_still_ahead` by t, t = 1..10:

| split | 1 | 2 | 3 | 4 | 5 | 6 | 7 | 8 | 9 | 10 |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| train | 1001 | 526 | 330 | 221 | 114 | 82 | 55 | 47 | 32 | 26 |
| valid | 239 | 123 | 79 | 48 | 26 | 20 | 14 | 11 | 8 | 7 |

Every run that submits gold also found it (submitted ⊂ found), in all splits. In train,
1,409 − 923 = **486 runs found gold but did not submit any**. The forced submit can
recover those, since BM25 ranks every observed paper. Earlier back-of-envelope figures
of 1,410 found and 922 submitted came from rounded percentages; the exact counts are
1,409 and 923.

**Parser decision.** `first_gold_step` matches with `build_dataset.ids_in_text` (every
bare `\d+` run). The template-aware parser in `src/features/hand.py`
(`observation_node_ids`) gives the same `first_gold_step` on all 3,000 runs
(`test_parsers_agree_on_first_gold_step`). `ids_in_text` is used so that labels do not
depend on code under `src/features/`, which may change for modelling reasons. A third
matcher, a whole-token regex `(?<!\d){id}(?!\d)` in
`test_first_gold_step_rederived_from_raw`, re-derives the label from raw JSON and agrees
on every run.

**Removed:** `label_observed_gold` (any gold id in any observation, all steps). It was
identical to `label_ever_finds_gold` on every trajectory in all three splits.

## What `accepted` measures

Exact-set match, not ranking quality:

| `reject_reason` | hit@1 | hit@5 | submitted precision |
| --- | --- | --- | --- |
| `valid` | 100% | 100% | 1.000 |
| `wrong_answer` | 46.6% | 70.4% | 0.395 |
| `invalid_graph_execution` | 43.3% | 57.2% | 0.401 |
| `no_submit_answer` | 12.0% | 33.5% | 0.000 |

Overall train hit@1 is ~41.5% against a 10.55% accept rate. Nearly half of
`wrong_answer` trajectories rank a gold node first.

`final_answer.ranked_node_ids` is produced by the **environment**, not the agent: BM25
over `observed_target_node_ids`, with submitted ids floated to the front (verified: the
ranked list begins with the submitted list verbatim in 420/421 cases). Trajectories that
submitted nothing still hit@1 14.6% of the time on BM25 alone. An hit@1 label therefore
partly scores BM25's luck.

## Known leaks

1. `len(trajectory) == 52` iff the run hit the turn cap — no exceptions in any split
   (see [Turn cap](#turn-cap)). In train this is exactly `no_submit_answer`.
2. Final step `valid_action is None` iff turn-capped — 582 / 151 / 172.
3. All 2000 trajectories end in `submit_answer`, including the 582 that never submitted
   (the last step is synthetic, appended during cleaning).
4. `observed_node_ids` / `observed_target_node_ids` aggregate the whole trajectory.
5. `subgraph.sampling.seed_ids` contains every gold node — 104/104 across a
   40-trajectory sample, including gold nodes the trajectory never observed. This cannot
   be a byproduct of seeding from visited nodes, since `no_submit_answer` trajectories
   observe only ~50% of their gold on average. Seeds also include nodes first observed
   late in the run (39/40 sampled), so they leak the future as well.
6. An empty observation appears only at step 51 of a turn-capped run (the unexecuted
   action at the cap). The builder drops it; the leak test asserts no prefix has one.
   Left in, `n_empty_observations` was the baseline's largest coefficient.
7. The final `submit_answer` observation is a JSON blob holding the whole-run aggregates
   (see [Final submit observation](#final-submit-observation)). The builder drops the
   final step; `test_no_submit_blob_in_prefix` asserts no prefix contains it.

## Sequence length

Serialising `query + think/action/observation` per step:

| prefix | median chars | ~tokens | p90 chars | ~tokens (p90) |
| --- | --- | --- | --- | --- |
| 1 step | 474 | 118 | 618 | 154 |
| 5 steps | 2,897 | 724 | 4,379 | 1,094 |
| 10 steps | 4,855 | 1,213 | 7,788 | 1,947 |
| 20 steps | 8,007 | 2,001 | 13,184 | 3,296 |

Single step: median 300 chars, p90 1,112, max 10,831. **512 tokens is exceeded around
step 4.** Options: truncate observations (BM25 score lists and `Displayed ... feature
keys` dumps are most of the bulk); hierarchical encoding (encode each step at ~75 tokens,
pool over step embeddings); ModernBERT (8k context); or last-N-step windowing.

## Early detectability

Mean duplicate-action rate within the first k steps:

| k | eventual looper | all others (surviving to k) |
| --- | --- | --- |
| 3 | 0.047 | 0.015 |
| 5 | 0.057 | 0.029 |
| 8 | 0.059 | 0.037 |
| 10 | 0.066 | 0.045 |

The separation is real but small early, and the right column is biased upward by
conditioning on survival. Duplicate-action count is the **baseline to beat**, not the
solution.
