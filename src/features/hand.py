"""Hand-built prefix features for the logistic-regression baseline."""

import re

from src.features import register

ACTION_TYPES = ("NodeFeature", "NeighbourCheck", "RetrieveNode", "NodeDegree")
RECENT_WINDOW = 3

RETRIEVED_LIST = re.compile(r"retrieved node IDs are: \[([^\]]*)\]")
QUOTED_ID = re.compile(r"'(\d+)'")
RANKED_NEIGHBOR = re.compile(r"\b(\d+) \(-?\d+\.\d+\)")
SUBJECT_NODE = re.compile(r"\bof (\d+) (?:is|are)\b")


def action_type(graph):
    return (graph or "").split("(", 1)[0].strip()


def observation_node_ids(observation):
    """Node ids named in an observation, by template: retrieved list, ranked
    neighbours, and the subject node of a feature/neighbour/degree reply."""
    text = observation or ""
    ids = set(SUBJECT_NODE.findall(text)) | set(RANKED_NEIGHBOR.findall(text))
    for block in RETRIEVED_LIST.findall(text):
        ids |= set(QUOTED_ID.findall(block))
    return {str(i) for i in ids}


def duplicate_action_rate(graphs):
    return 1 - len(set(graphs)) / len(graphs)


def repeats_earlier(graphs, i):
    return graphs[i] in graphs[:i]


def recent_repeat_rate(graphs, window):
    recent = range(max(0, len(graphs) - window), len(graphs))
    return sum(repeats_earlier(graphs, i) for i in recent) / len(recent)


def node_discovery(observations):
    """(n_distinct_nodes, steps_since_new_node). Steps with no new node since
    the start count from step 0."""
    seen, last_new = set(), 0
    for step_number, obs in enumerate(observations, start=1):
        ids = observation_node_ids(obs)
        if ids - seen:
            last_new = step_number
        seen |= ids
    return len(seen), len(observations) - last_new


@register("hand")
def hand_features(inputs):
    steps = inputs["steps"]
    graphs = [s["graph"] or "" for s in steps]
    observations = [s["observation"] or "" for s in steps]
    actions = [action_type(g) for g in graphs]
    n_distinct_nodes, steps_since_new_node = node_discovery(observations)
    features = {
        "t": inputs["t"],
        "duplicate_action_rate": duplicate_action_rate(graphs),
        "n_invalid": sum(s["valid_action"] == 0 for s in steps),
        "n_distinct_nodes": n_distinct_nodes,
        "mean_observation_length": sum(map(len, observations)) / len(observations),
        "steps_since_new_node": steps_since_new_node,
        "recent_repeat_rate": recent_repeat_rate(graphs, RECENT_WINDOW),
    }
    for a in ACTION_TYPES:
        features[f"n_{a}"] = actions.count(a)
    return features
