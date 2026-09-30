"""Featurizer registry.

A featurizer maps `model_inputs(row)` — {t, query, steps}, never the full row — to a
flat feature dict. Register it with `@register("name")` where it is defined, in a
module under src/features/. Every module in this package is imported below, so
every registered featurizer is leak-checked by tests/test_no_leakage.py.

Call featurizers only through `featurize(name, row)`.
"""

import importlib
import pkgutil

from src.data.build_dataset import model_inputs

FEATURIZERS = {}


def featurize(name, row):
    return FEATURIZERS[name](model_inputs(row))


def register(name):
    def decorator(fn):
        assert name not in FEATURIZERS, f"featurizer {name!r} registered twice"
        FEATURIZERS[name] = fn
        return fn
    return decorator


def _import_all():
    for info in sorted(pkgutil.iter_modules(__path__), key=lambda m: m.name):
        importlib.import_module(f"{__name__}.{info.name}")


_import_all()
