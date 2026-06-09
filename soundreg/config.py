"""Minimal YAML config system: nested dicts with attribute access,
``_base_`` inheritance, and dotted-key CLI overrides.

The YAML files are the schema — adding a field means adding a line to a
config, nothing else.
"""

from __future__ import annotations

import copy
from pathlib import Path

import yaml


class Cfg(dict):
    """Dict with recursive attribute access: cfg.model.encoder.name."""

    def __getattr__(self, key):
        try:
            return self[key]
        except KeyError as e:
            raise AttributeError(key) from e

    def __setattr__(self, key, value):
        self[key] = value

    @staticmethod
    def from_dict(d: dict) -> "Cfg":
        out = Cfg()
        for k, v in d.items():
            out[k] = Cfg.from_dict(v) if isinstance(v, dict) else v
        return out

    def to_dict(self) -> dict:
        return {k: v.to_dict() if isinstance(v, Cfg) else v for k, v in self.items()}

    def get_path(self, dotted: str, default=None):
        node = self
        for part in dotted.split("."):
            if not isinstance(node, dict) or part not in node:
                return default
            node = node[part]
        return node


def _deep_merge(base: dict, override: dict) -> dict:
    out = copy.deepcopy(base)
    for k, v in override.items():
        if k in out and isinstance(out[k], dict) and isinstance(v, dict):
            out[k] = _deep_merge(out[k], v)
        else:
            out[k] = copy.deepcopy(v)
    return out


def _load_yaml_with_bases(path: Path) -> dict:
    with open(path) as f:
        raw = yaml.safe_load(f) or {}
    base_rel = raw.pop("_base_", None)
    if base_rel is not None:
        base = _load_yaml_with_bases((path.parent / base_rel).resolve())
        raw = _deep_merge(base, raw)
    return raw


def apply_override(cfg: dict, dotted_key: str, value_str: str) -> None:
    """Set ``a.b.c=value`` parsing the value as YAML (so 1e-3, true, [1,2] work)."""
    node = cfg
    parts = dotted_key.split(".")
    for part in parts[:-1]:
        if part not in node or not isinstance(node[part], dict):
            node[part] = Cfg()
        node = node[part]
    node[parts[-1]] = yaml.safe_load(value_str)


def load_config(path: str | Path, overrides: list[str] | None = None) -> Cfg:
    raw = _load_yaml_with_bases(Path(path).resolve())
    cfg = Cfg.from_dict(raw)
    for ov in overrides or []:
        key, _, val = ov.partition("=")
        apply_override(cfg, key.strip(), val.strip())
    return cfg


def save_config(cfg: Cfg, path: str | Path) -> None:
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w") as f:
        yaml.safe_dump(cfg.to_dict(), f, sort_keys=False)
