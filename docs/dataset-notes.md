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

1. `len(trajectory) == 52` iff `no_submit_answer` — no exceptions.
2. Final step `valid_action is None` iff `no_submit_answer` — 582/582.
3. All 2000 trajectories end in `submit_answer`, including the 582 that never submitted
   (the last step is synthetic, appended during cleaning).
4. `observed_node_ids` / `observed_target_node_ids` aggregate the whole trajectory.
5. `subgraph.sampling.seed_ids` contains every gold node — 104/104 across a
   40-trajectory sample, including gold nodes the trajectory never observed. This cannot
   be a byproduct of seeding from visited nodes, since `no_submit_answer` trajectories
   observe only ~50% of their gold on average. Seeds also include nodes first observed
   late in the run (39/40 sampled), so they leak the future as well.

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
