# graphrag-early-exit

Predict, from a **prefix** of a GraphRAG search trajectory, whether that search is on a
path that will fail — so the agent can exit early instead of burning its turn budget.

Undergraduate research project, KIND Lab (PI: Dr. Yu Wang), University of Georgia.
Data: 2000 / 500 / 500 MAG GraphDancer trajectories over STaRK-QA queries. Each
trajectory is a ReAct-style loop of `think` -> `graph` (tool call) -> `observation`,
capped at 50 turns.

A **prefix** is the query plus every (think, graph, observation) step from step 1
through step `t` — exactly what the agent had seen when choosing its next action, and
exactly what a deployed classifier would see. One training instance = one prefix plus
its trajectory's outcome labels.

## Read before writing code

- `docs/dataset-notes.md` — schema, verified statistics, and every known leak.
  **Read it before writing or modifying anything under `src/data/` or `src/eval/`.**

## Commands

```bash
source .venv/bin/activate          # or: uv run <cmd>
python -m src.data.build_dataset   # raw trajectories -> data/instances_{split}.jsonl
python -m src.train.baseline       # hand-feature logistic regression
python -m src.train.bert           # fine-tune the encoder
pytest tests/ -q                   # MUST pass before any commit
```

Data lives at `$MAG_DATA` (default `~/data/mag-trajectories`), never in the repo.
Built instances go to `data/`, which is gitignored.

## Hard constraints

These are correctness rules, not style preferences. Violating one produces a model
that scores well and means nothing, with no error raised.

### 1. Never use these as model input

Labels and oracle fields:
- `ground_truth_answer_ids`
- `quality` (both `accepted` and `reject_reason`)
- `shortest_path`
- `final_answer` (including `ranked_node_ids`, `submitted_node_ids`)

Trajectory-level aggregates — these summarise the **whole** trajectory, so at prefix
`t` they describe steps that have not happened yet:
- `observed_node_ids`
- `observed_target_node_ids`

Subgraph fields — `sampling.seed_ids` and `verification.trajectory_node_ids` contain
**every gold node** (verified: 104/104 across a 40-trajectory sample), including gold
nodes the trajectory never observed:
- `subgraph["sampling"]`
- `subgraph["verification"]`
- `subgraph["nodes"]` / `subgraph["edges"]` — softer, but the sample was *constructed*
  from the answers, so gold presence is guaranteed while other nodes arrive only by
  2-hop expansion. Do not use without explicit sign-off.

Derived quantities that leak the same information:
- `len(trajectory)` — exactly 52 iff the run hit the turn cap, no exceptions in any split.
  In train this coincides exactly with `reject_reason == "no_submit_answer"` (582/582);
  valid qid 2186 (`timeout`) and test qid 1122 (`missing_think_graph_format`) also hit
  the cap
- the final step's `valid_action is None` — same tell, iff turn-capped in every split
  (582 / 151 / 172)
- "is this the last step", "total steps", or any feature computed over the full trajectory

### 2. Train on prefixes, never whole trajectories

One training instance = one `(trajectory, prefix ending at step t)` pair. A prefix may
only contain `trajectory[0..t]`. Never read `trajectory[i]` for `i > t`.

Drop the **final** step of every trajectory: it is always `submit_answer`, either
terminal or the synthetic step appended during cleaning, and no exit decision remains
after it. Keep mid-run `submit_answer` calls (failed, `valid_action=0`) as ordinary
steps — the agent took them and the trajectory continued.

Then also drop a **trailing step with an empty observation**. It is the unexecuted
action at the 50-turn cap, a termination artifact like `valid_action=None`. Only
turn-capped runs have one, always at step 51: every `no_submit_answer` run, plus test
qid 1122. Valid qid 2186 hit the cap too, but its step 51 is a malformed-output error
message rather than empty, so it keeps all 51 prefixes. Empty observations never occur
mid-run, so no instance may contain one. Every instance must satisfy `len(steps) == t`.

Expected instance counts: train 39,679 · valid 9,960 · test 10,893.

### 3. Target and labels

**Target:** will the search observe at least one gold node? "Observed" means a gold id
appears in the observation of a **kept** step (one that appears in prefixes; see §2).
Early exit is a **forced submit**: BM25 over every observed paper node, submitted nodes
first. So exiting at step `t` costs only a *first* gold sighting that would have come
after `t`.

Keep three layers distinct:
1. **Labels** — deterministic, computed by the builder from the full trajectory and gold.
2. **Model scores** — learned from `model_inputs(row)` only.
3. **Policy** — a rule on scores (warmup, stride, threshold, patience), evaluated by replay.

| Column | Level | Definition | Positive (trajectories, train) | Positive (instances, train) |
| --- | --- | --- | --- | --- |
| `label_first_gold_step` | trajectory, oracle metadata | smallest kept `step` whose observation contains a gold id, else null | median 2, p90 5, max 40 | — |
| `label_ever_finds_gold` | trajectory | `label_first_gold_step is not None` | 1,409 (70.45%) | 59.0% |
| `label_gold_still_ahead` | prefix | `label_ever_finds_gold and t < label_first_gold_step` | — | 2,580 (6.5%) |
| `label_submits_gold` | trajectory | `set(map(str, final_answer.submitted_node_ids))` ∩ gold ≠ ∅ | 923 (46.15%) | 14.9% |
| `label_hits_cap` | trajectory | ran to the turn cap: raw `len(trajectory) == 52` | 582 (29.1%) | 73.3% |
| `label_accepted` | trajectory | `quality.accepted`, exact-set match | 211 (10.55%) | 3.2% |
| `label_hit1` | trajectory | `final_answer.ranked_node_ids[0]` in gold | 41.5% | 21.3% |

Instance-level rates differ from trajectory-level because long trajectories contribute
more prefixes — a turn-capped run contributes 50. `python -m src.data.build_dataset`
prints every number in this table.

**Every `label_*` column is oracle information and never model input** — including
`label_first_gold_step`. Labels may read gold ids and the full trajectory; `model_inputs`
and featurizers may not. The leak tests deny any key starting with `label_`.

First-sighting matching lives in one helper, `build_dataset.first_gold_step`, using
`ids_in_text`. `label_submits_gold` implies `label_ever_finds_gold` (submitted ⊂ found).
`label_hits_cap` targets wasted compute; it equals `reject_reason == "no_submit_answer"`
except for valid qid 2186 and test qid 1122.

Which label the model trains on is still **unconfirmed by Dr. Wang**. Never hardcode
one: `src.train.baseline` takes `--label` as a **required** argument, with no default,
restricted to `BINARY_LABELS`.

### 4. Node IDs are inconsistently typed

`str()` both sides before any comparison or set membership test. Ints and strings both
appear across fields, and a mismatch silently yields zero matches rather than an error.

### 5. Evaluation must mirror deployment

- Report **PR-AUC** and **ROC-AUC** together with the instance-level positive rate.
  Never report accuracy. Random PR-AUC equals the positive rate — on valid, 0.619 for
  `label_ever_finds_gold`, 0.062 for `label_gold_still_ahead`, 0.758 for
  `label_hits_cap` — so read PR-AUC as lift over that.
- Also report both metrics on the **early subset (t ≤ 10)**. Late prefixes of looping
  runs are trivially easy and dominate full-set numbers; early prefixes are where
  exiting actually saves compute.
- The headline result is **steps saved vs. correct answers lost**, computed by replaying
  an exit policy (warmup, stride `k`, threshold, patience) over logged trajectories.
- Whatever policy the replay uses must be the policy being reported.
- Evaluate on valid. Do not touch test until final results.

## Conventions

- Python 3.12, `uv` for dependencies. Do not add a dependency without asking.
- Seed everything. Use `sorted()` on every glob — filesystem order is not stable.
- Every number that appears in a commit message, report, or email to the PI must be
  reproducible by a script in this repo.
- Prefer small functions with explicit arguments over classes holding state.
- Do not create files that were not asked for. No README unless requested.

## Leak test

`tests/test_no_leakage.py` asserts that no forbidden key appears in built features,
that no prefix contains data from steps beyond its cutoff, that no instance contains
its trajectory's final step, and that `len(steps) == t`. **Any new feature must pass
it.** If a change requires editing that test, stop and ask first — that is the signal
that something is being rationalised.

`tests/test_labels.py` checks the label definitions against an independent re-derivation
from raw data, and that no `label_*` key reaches `model_inputs` or any featurizer.

## Open questions (awaiting Dr. Wang)

1. Which label is the target.
2. Whether to predict all failure, or non-termination specifically.
3. Whether subgraphs are intended as model input at all.
4. Scoring cadence at inference (every step vs. every k).

Until answered: text-only input, per-step scoring, working target "will the search
observe ≥ 1 gold node" (§3), with `--label` always passed explicitly.