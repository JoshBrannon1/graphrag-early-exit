"""Leak tests for built prefix instances and every featurizer.

Requires `python -m src.data.build_dataset` to have been run. Missing files fail,
they do not skip.
"""

import json
import random
import re
from functools import cache
from pathlib import Path

import pytest

from src.data.build_dataset import DEFAULT_OUT_DIR, DEFAULT_RAW_DIR, STEP_KEYS, model_inputs
from src.features import FEATURIZERS, featurize

SPLITS = ("train", "valid", "test")
SAMPLE_PER_SPLIT = 100
SEED = 0
TEXT_FIELDS = ("think", "graph", "observation")
VALID_ACTIONS = {0, 1}
# Hardcoded on purpose: widening model_inputs or STEP_KEYS in src/ must fail here.
ALLOWED_INPUT_KEYS = {"t", "query", "steps"}
ALLOWED_STEP_KEYS = {"step", "think", "graph", "observation", "valid_action"}
REPO_ROOT = Path(__file__).resolve().parents[1]
REGISTRY_MODULE = REPO_ROOT / "src" / "features" / "__init__.py"
DIRECT_REGISTRY_ACCESS = re.compile(r"FEATURIZERS\s*(\[|\.get\b|\.values\b|\.items\b)")
FEATURES_DIR = REPO_ROOT / "src" / "features"
FEATURE_IO = re.compile(
    r"\bopen\s*\(|\bPath\s*\(|\bread_text\b|\bread_bytes\b|\bjson\.load\s*\("
    r"|\bos\.environ\b|\bos\.getenv\b|\bDEFAULT_RAW_DIR\b|\bDEFAULT_OUT_DIR\b"
)
N_SUBGRAPH_FILES = 5

BASE_FORBIDDEN = {
    "ground_truth_answer_ids",
    "quality",
    "shortest_path",
    "final_answer",
    "observed_node_ids",
    "observed_target_node_ids",
    "subgraph",
}


def instance_path(split):
    path = DEFAULT_OUT_DIR / f"instances_{split}.jsonl"
    assert path.exists(), f"{path} missing — run python -m src.data.build_dataset"
    return path


def iter_rows(split):
    with instance_path(split).open() as f:
        for line in f:
            yield json.loads(line)


def sample_rows(split, n, seed):
    with instance_path(split).open("rb") as f:
        n_lines = sum(1 for _ in f)
    chosen = set(random.Random(f"{seed}-{split}").sample(range(n_lines), n))
    with instance_path(split).open() as f:
        return [json.loads(line) for i, line in enumerate(f) if i in chosen]


@cache
def raw_trajectory(split, qid):
    path = DEFAULT_RAW_DIR / split / f"q_{qid}.json"
    return json.loads(path.read_text())["trajectory"]


def walk_keys(obj, path=""):
    """Yield (path, key) for every dict key at any depth."""
    if isinstance(obj, dict):
        for k, v in obj.items():
            yield f"{path}.{k}", k
            yield from walk_keys(v, f"{path}.{k}")
    elif isinstance(obj, list):
        for i, v in enumerate(obj):
            yield from walk_keys(v, f"{path}[{i}]")


@cache
def subgraph_keys():
    paths = sorted((DEFAULT_RAW_DIR / "subgraphs" / "train").glob("*/subgraph.json"))
    assert paths, f"no subgraph files under {DEFAULT_RAW_DIR / 'subgraphs' / 'train'}"
    keys = set()
    for p in paths[:N_SUBGRAPH_FILES]:
        keys |= {k for _, k in walk_keys(json.loads(p.read_text()))}
    return frozenset(keys)


def forbidden_keys():
    return BASE_FORBIDDEN | subgraph_keys()


def forbidden_hits(features, forbidden):
    return [(path, key) for path, key in walk_keys(features)
            if key in forbidden or str(key).startswith("label_")]


def future_text_leaks(row, trajectory):
    """Future-step texts that appear in the instance's steps.

    A future text that also occurs in the true prefix (or the query) is skipped:
    looping runs repeat actions verbatim, so its presence proves nothing.
    Returns (leaks, n_checked).
    """
    t = row["t"]
    legit = "\x00".join([row["query"]] + [s[f] or "" for s in trajectory[:t] for f in TEXT_FIELDS])
    present = "\x00".join(s[f] or "" for s in row["steps"] for f in TEXT_FIELDS)
    leaks, n_checked = [], 0
    for s in trajectory[t:]:
        for field in TEXT_FIELDS:
            text = s[field]
            if not text or text in legit:
                continue
            n_checked += 1
            if text in present:
                leaks.append((s["step"], field, text[:80]))
    return leaks, n_checked


@pytest.fixture(scope="module")
def sample():
    rows = [r for split in SPLITS for r in sample_rows(split, SAMPLE_PER_SPLIT, SEED)]
    assert len(rows) >= 200
    return rows


def test_registry_is_populated():
    assert FEATURIZERS, "src.features registered no featurizers"


def test_model_inputs_allowlist(sample):
    assert set(STEP_KEYS) == ALLOWED_STEP_KEYS, f"STEP_KEYS changed: {STEP_KEYS}"
    for row in sample:
        inputs = model_inputs(row)
        where = f"{row['split']} qid={row['qid']} t={row['t']}"
        assert set(inputs) == ALLOWED_INPUT_KEYS, f"{where}: model_inputs keys {sorted(inputs)}"
        assert isinstance(inputs["t"], int), f"{where}: t is {type(inputs['t']).__name__}"
        assert isinstance(inputs["query"], str), f"{where}: query is {type(inputs['query']).__name__}"
        for s in inputs["steps"]:
            assert set(s) == ALLOWED_STEP_KEYS, f"{where} step={s.get('step')}: keys {sorted(s)}"


def test_featurizers_called_only_through_featurize():
    offenders = []
    for path in sorted(REPO_ROOT.glob("src/**/*.py")) + sorted(REPO_ROOT.glob("tests/**/*.py")):
        if path == REGISTRY_MODULE:
            continue
        for lineno, line in enumerate(path.read_text().splitlines(), start=1):
            if DIRECT_REGISTRY_ACCESS.search(line):
                offenders.append(f"{path.relative_to(REPO_ROOT)}:{lineno}: {line.strip()}")
    assert not offenders, "call featurizers via featurize(name, row) only:\n" + "\n".join(offenders)


def test_featurizers_do_no_io():
    """Featurizers may see only their argument: no file or environment access."""
    offenders = []
    for path in sorted(FEATURES_DIR.glob("**/*.py")):
        for lineno, line in enumerate(path.read_text().splitlines(), start=1):
            if not line.strip().startswith("#") and FEATURE_IO.search(line):
                offenders.append(f"{path.relative_to(REPO_ROOT)}:{lineno}: {line.strip()}")
    assert not offenders, "file/env I/O in src/features/:\n" + "\n".join(offenders)


def test_no_forbidden_keys_in_model_inputs(sample):
    forbidden = forbidden_keys()
    for row in sample:
        hits = forbidden_hits(model_inputs(row), forbidden)
        assert not hits, (
            f"model_inputs leaked forbidden key(s) for "
            f"{row['split']} qid={row['qid']} t={row['t']}: {hits[:5]}"
        )


@pytest.mark.parametrize("name", sorted(FEATURIZERS))
def test_no_forbidden_keys_in_features(sample, name):
    forbidden = forbidden_keys()
    for row in sample:
        hits = forbidden_hits(featurize(name, row), forbidden)
        assert not hits, (
            f"featurizer {name!r} leaked forbidden key(s) for "
            f"{row['split']} qid={row['qid']} t={row['t']}: {hits[:5]}"
        )


def test_no_future_text_in_prefix(sample):
    total_checked = 0
    for row in sample:
        leaks, n_checked = future_text_leaks(row, raw_trajectory(row["split"], row["qid"]))
        total_checked += n_checked
        assert not leaks, (
            f"future text in {row['split']} qid={row['qid']} t={row['t']}: "
            f"(step, field, text) = {leaks[:3]}"
        )
    assert total_checked > 0, "no future text was checkable — test is vacuous"


@pytest.mark.parametrize("split", SPLITS)
def test_prefix_structure(split):
    """Every instance: len(steps) == t, steps are 1..t, final step excluded,
    every valid_action in {0, 1} (None only ever marks the final step), and no
    empty observation (only the unexecuted turn-cap step has one)."""
    for row in iter_rows(split):
        qid, t, steps = row["qid"], row["t"], row["steps"]
        trajectory = raw_trajectory(split, qid)
        assert len(steps) == t, f"{split} qid={qid} t={t}: len(steps)={len(steps)}"
        assert [s["step"] for s in steps] == list(range(1, t + 1)), (
            f"{split} qid={qid} t={t}: step numbers {[s['step'] for s in steps]}"
        )
        assert t < len(trajectory), (
            f"{split} qid={qid} t={t}: prefix reaches final step {len(trajectory)}"
        )
        bad = [(s["step"], s["valid_action"]) for s in steps if s["valid_action"] not in VALID_ACTIONS]
        assert not bad, f"{split} qid={qid} t={t}: (step, valid_action) outside {{0, 1}}: {bad[:5]}"
        empty = [s["step"] for s in steps if not (s["observation"] or "").strip()]
        assert not empty, f"{split} qid={qid} t={t}: empty observation at step(s) {empty}"
        final = {k: trajectory[-1][k] for k in STEP_KEYS}
        assert final not in steps, f"{split} qid={qid} t={t}: contains the final step"
    raw_trajectory.cache_clear()


def plant_step(row, step):
    return dict(row, steps=row["steps"] + [{k: step[k] for k in STEP_KEYS}])


def next_step_checkable(row):
    """True if trajectory[t] has text the future-text detector can check."""
    trajectory = raw_trajectory(row["split"], row["qid"])
    _, n_checked = future_text_leaks(dict(row, steps=[]), trajectory[: row["t"] + 1])
    return n_checked > 0


def test_detectors_catch_planted_leaks(sample):
    row = next(r for r in sample if r["t"] >= 2)
    trajectory = raw_trajectory(row["split"], row["qid"])
    leaks, _ = future_text_leaks(plant_step(row, trajectory[-1]), trajectory)
    assert leaks, f"future-text detector missed a planted final step (qid={row['qid']})"

    row = next((r for r in sample if next_step_checkable(r)), None)
    assert row is not None, "no sampled row has a checkable trajectory[t]"
    trajectory = raw_trajectory(row["split"], row["qid"])
    leaks, _ = future_text_leaks(plant_step(row, trajectory[row["t"]]), trajectory)
    assert any(step == row["t"] + 1 for step, _, _ in leaks), (
        f"future-text detector missed planted trajectory[t] (qid={row['qid']} t={row['t']})"
    )

    features = dict(model_inputs(row), nested={"seed_ids": [], "label_looping": True})
    hits = {k for _, k in forbidden_hits(features, forbidden_keys())}
    assert hits == {"seed_ids", "label_looping"}, f"forbidden-key detector returned {hits}"
