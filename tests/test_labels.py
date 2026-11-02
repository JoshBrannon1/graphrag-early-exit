"""Label tests for built prefix instances.

Requires `python -m src.data.build_dataset` to have been run. Missing files fail,
they do not skip. Each check is a function returning a list of problems, so the
plants at the bottom can show the same checks catch a violation.
"""

import json
import random
import re
from collections import defaultdict
from functools import cache

import pytest

import src.data.build_dataset as bd
from src.features import FEATURIZERS, featurize
from src.features.hand import observation_node_ids

SPLITS = ("train", "valid", "test")
SAMPLE_PER_SPLIT = 100
SEED = 0
# Constant across every prefix of a run.
RUN_LABELS = ("label_ever_finds_gold", "label_submits_gold", "label_hits_cap", "label_first_gold_step")
# Keys of the environment's final-submit JSON; no prefix observation may contain them.
SUBMIT_BLOB_KEYS = ("ranking_method", "candidate_source", "observed_target_node_ids")
MID_RUN_SUBMIT_ERROR = "Invalid submit_answer arguments."
MAX_SHOWN = 5


def instance_path(split):
    path = bd.DEFAULT_OUT_DIR / f"instances_{split}.jsonl"
    assert path.exists(), f"{path} missing — run python -m src.data.build_dataset"
    return path


def iter_rows(split):
    with instance_path(split).open() as f:
        for line in f:
            yield json.loads(line)


def slim(row):
    """Row without steps, except the last observation and whether any observation
    names a gold id — kept only at t in {first_gold_step - 1, first_gold_step}."""
    out = {k: v for k, v in row.items() if k != "steps"}
    first = row["label_first_gold_step"]
    if first is not None and row["t"] in (first - 1, first):
        out["observations"] = [s["observation"] or "" for s in row["steps"]]
    return out


@cache
def label_rows(split):
    return tuple(slim(r) for r in iter_rows(split))


@cache
def raw_trajectories(split):
    paths = sorted((bd.DEFAULT_RAW_DIR / split).glob("q_*.json"))
    assert paths, f"no raw trajectories under {bd.DEFAULT_RAW_DIR / split}"
    return {str(t["qid"]): t for t in (json.loads(p.read_text()) for p in paths)}


def by_run(rows):
    runs = defaultdict(list)
    for r in rows:
        runs[(r["split"], r["qid"])].append(r)
    return {k: sorted(v, key=lambda r: r["t"]) for k, v in runs.items()}


def names_gold(text, gold):
    """True if text names any gold id as a whole digit token. Deliberately not
    bd.ids_in_text, so the label is checked by a second matcher."""
    return any(re.search(rf"(?<!\d){re.escape(g)}(?!\d)", text) for g in gold)


# ---- checks: each returns a list of human-readable problems -------------------------

def inconstant_run_labels(rows):
    return [f"{split} qid={qid}: {key} takes values {sorted({r[key] for r in rs}, key=str)}"
            for (split, qid), rs in by_run(rows).items()
            for key in RUN_LABELS if len({r[key] for r in rs}) > 1]


def nonmonotone_still_ahead(rows):
    problems = []
    for (split, qid), rs in by_run(rows).items():
        seq = [r["label_gold_still_ahead"] for r in rs]
        flips = [rs[i]["t"] for i in range(1, len(seq)) if not seq[i - 1] and seq[i]]
        if flips:
            problems.append(f"{split} qid={qid}: still_ahead goes False->True at t={flips}")
    return problems


def still_ahead_mismatches(rows):
    problems = []
    for r in rows:
        first = r["label_first_gold_step"]
        expected = first is not None and r["t"] < first
        if r["label_gold_still_ahead"] != expected:
            problems.append(f"{r['split']} qid={r['qid']} t={r['t']}: still_ahead="
                            f"{r['label_gold_still_ahead']} but first_gold_step={first}")
    return problems


def still_ahead_count_mismatches(rows):
    problems = []
    for (split, qid), rs in by_run(rows).items():
        first = rs[0]["label_first_gold_step"]
        expected = first - 1 if first is not None else 0
        got = sum(r["label_gold_still_ahead"] for r in rs)
        if got != expected:
            problems.append(f"{split} qid={qid}: {got} still_ahead rows, expected {expected} "
                            f"(first_gold_step={first})")
    return problems


def still_ahead_without_finding(rows):
    return [f"{r['split']} qid={r['qid']} t={r['t']}: still_ahead without ever_finds_gold"
            for r in rows if r["label_gold_still_ahead"] and not r["label_ever_finds_gold"]]


def submits_without_finding(rows):
    return [f"{r['split']} qid={r['qid']}: submits_gold but not ever_finds_gold"
            for r in rows if r["t"] == 1 and r["label_submits_gold"] and not r["label_ever_finds_gold"]]


def rederived_first_gold_step(traj):
    """Independent of bd.first_gold_step / bd.cutoffs / bd.ids_in_text."""
    steps = traj["trajectory"][:-1]
    if steps and not (steps[-1]["observation"] or "").strip():
        steps = steps[:-1]
    gold = {str(g) for g in traj["ground_truth_answer_ids"]}
    for s in steps:
        if names_gold(s["observation"] or "", gold):
            return s["step"]
    return None


def first_gold_step_mismatches(rows, trajs):
    problems = []
    for (split, qid), rs in by_run(rows).items():
        expected = rederived_first_gold_step(trajs[qid])
        if rs[0]["label_first_gold_step"] != expected:
            problems.append(f"{split} qid={qid}: label_first_gold_step="
                            f"{rs[0]['label_first_gold_step']}, re-derived {expected}")
    return problems


def sighting_text_mismatches(rows, trajs):
    problems = []
    for r in rows:
        if "observations" not in r:
            continue
        gold = {str(g) for g in trajs[r["qid"]]["ground_truth_answer_ids"]}
        where = f"{r['split']} qid={r['qid']} t={r['t']}"
        if r["t"] == r["label_first_gold_step"] and not names_gold(r["observations"][-1], gold):
            problems.append(f"{where}: t == first_gold_step but last observation names no gold id")
        if r["t"] == r["label_first_gold_step"] - 1:
            hits = [i + 1 for i, o in enumerate(r["observations"]) if names_gold(o, gold)]
            if hits:
                problems.append(f"{where}: t == first_gold_step - 1 but step(s) {hits} name gold")
    return problems


def submit_blob_observations(rows):
    return [f"{r['split']} qid={r['qid']} t={r['t']} step={s['step']}: observation contains {keys}"
            for r in rows for s in r["steps"]
            if (keys := [k for k in SUBMIT_BLOB_KEYS if k in (s["observation"] or "")])]


def label_key_hits(row, featurizer_names):
    def label_keys(d):
        return sorted(k for k in d if str(k).startswith("label_"))
    hits = [f"model_inputs: {k}" for k in label_keys(bd.model_inputs(row))]
    hits += [f"featurizer {n!r}: {k}" for n in featurizer_names for k in label_keys(featurize(n, row))]
    return hits


def fail_message(problems):
    return f"{len(problems)} problem(s):\n  " + "\n  ".join(problems[:MAX_SHOWN])


# ---- fixtures ------------------------------------------------------------------------

@pytest.fixture(scope="module")
def rows():
    return [r for split in SPLITS for r in label_rows(split)]


@pytest.fixture(scope="module")
def sample():
    out = []
    for split in SPLITS:
        n_lines = len(label_rows(split))
        chosen = set(random.Random(f"{SEED}-{split}").sample(range(n_lines), SAMPLE_PER_SPLIT))
        out += [r for i, r in enumerate(iter_rows(split)) if i in chosen]
    return out


# ---- tests ---------------------------------------------------------------------------

def test_run_labels_constant(rows):
    problems = inconstant_run_labels(rows)
    assert not problems, fail_message(problems)


def test_still_ahead_monotone(rows):
    problems = nonmonotone_still_ahead(rows)
    assert not problems, fail_message(problems)


def test_still_ahead_definition(rows):
    problems = still_ahead_mismatches(rows)
    assert not problems, fail_message(problems)


def test_still_ahead_count_per_run(rows):
    problems = still_ahead_count_mismatches(rows)
    assert not problems, fail_message(problems)


def test_no_still_ahead_without_finding(rows):
    problems = still_ahead_without_finding(rows)
    assert not problems, fail_message(problems)


def test_submits_gold_implies_finds_gold(rows):
    problems = submits_without_finding(rows)
    assert not problems, fail_message(problems)


@pytest.mark.parametrize("split", SPLITS)
def test_first_gold_step_rederived_from_raw(split):
    problems = first_gold_step_mismatches(label_rows(split), raw_trajectories(split))
    assert not problems, fail_message(problems)
    assert len(by_run(label_rows(split))) == len(raw_trajectories(split))


@pytest.mark.parametrize("split", SPLITS)
def test_first_gold_step_matches_prefix_text(split):
    checked = [r for r in label_rows(split) if "observations" in r]
    assert checked, f"{split}: no rows at first_gold_step — test is vacuous"
    problems = sighting_text_mismatches(checked, raw_trajectories(split))
    assert not problems, fail_message(problems)


@pytest.mark.parametrize("split", SPLITS)
def test_no_submit_blob_in_prefix(split):
    problems = submit_blob_observations(iter_rows(split))
    assert not problems, fail_message(problems)


@pytest.mark.parametrize("split", SPLITS)
def test_parsers_agree_on_first_gold_step(split):
    """bd.ids_in_text (bare digits) and the template parser give the same first sighting."""
    problems = []
    for qid, traj in raw_trajectories(split).items():
        gold = bd.gold_ids(traj)
        kept = traj["trajectory"][: max(bd.cutoffs(traj["trajectory"]), default=0)]
        template = next((s["step"] for s in kept if gold & observation_node_ids(s["observation"])), None)
        if template != bd.first_gold_step(traj):
            problems.append(f"{split} qid={qid}: ids_in_text {bd.first_gold_step(traj)}, template {template}")
    assert not problems, fail_message(problems)


def test_mid_run_submits_are_error_strings():
    """Mid-run submit_answer steps are kept as ordinary steps; their observation is
    the environment's argument error, never the final-submit blob."""
    steps = [(split, qid, s) for split in SPLITS for qid, traj in raw_trajectories(split).items()
             for s in traj["trajectory"][:-1] if bd.is_submit(s)]
    assert steps, "no mid-run submit_answer steps — test is vacuous"
    problems = [f"{split} qid={qid} step={s['step']}: {s['observation'][:80]!r}" for split, qid, s in steps
                if not (s["observation"] or "").startswith(MID_RUN_SUBMIT_ERROR)]
    assert not problems, fail_message(problems)


def test_no_label_keys_in_inputs_or_features(sample):
    names = sorted(FEATURIZERS)
    assert names, "src.features registered no featurizers"
    problems = [f"{r['split']} qid={r['qid']} t={r['t']}: {h}"
                for r in sample for h in label_key_hits(r, names)]
    assert not problems, fail_message(problems)


# ---- planted violations: the checks above must catch them ---------------------------

def test_checks_catch_planted_violations(rows, sample, monkeypatch):
    # t = first_gold_step + 1 flipped to True: True..True, False, True.
    finding = next(r for r in rows if (r["label_first_gold_step"] or 0) >= 2
                   and r["t"] == r["label_first_gold_step"] + 1)
    run = [r for r in rows if (r["split"], r["qid"]) == (finding["split"], finding["qid"])]
    flipped = [dict(r, label_gold_still_ahead=True) if r["t"] == finding["t"] else r for r in run]
    assert nonmonotone_still_ahead(flipped), "monotonicity check missed a planted flip"
    assert still_ahead_mismatches(flipped), "definition check missed a planted flip"

    no_find = next(r for r in rows if r["t"] == 1 and not r["label_ever_finds_gold"])
    assert submits_without_finding([dict(no_find, label_submits_gold=True)]), (
        "submits_gold check missed a planted submits-without-finding run"
    )

    row = sample[0]
    blob = raw_trajectories(row["split"])[row["qid"]]["trajectory"][-1]["observation"]
    planted = dict(row, steps=row["steps"][:-1] + [dict(row["steps"][-1], observation=blob)])
    assert submit_blob_observations([planted]), "blob check missed a planted final-submit blob"

    real = bd.model_inputs
    monkeypatch.setattr(bd, "model_inputs", lambda r: dict(real(r), label_gold_still_ahead=True))
    assert label_key_hits(row, []) == ["model_inputs: label_gold_still_ahead"], (
        "label-key check missed label_gold_still_ahead planted in model_inputs"
    )
