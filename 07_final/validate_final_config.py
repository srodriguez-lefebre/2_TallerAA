#!/usr/bin/env python3
"""Validate that the deliverable pipeline clones the selected e100 3-way model."""

from __future__ import annotations

import ast
import hashlib
import json
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd


ROOT = Path(__file__).resolve().parents[1]
PIPELINE_NOTEBOOK = ROOT / "100. Entregable" / "00_pipeline_entregable.ipynb"
FINAL_SOURCE = (
    ROOT
    / "investigation"
    / "results"
    / "submissions"
    / "parallel100_20260702"
    / "e100_headsep25_globalmel375_f1024_375.csv"
)
EXPECTED_FINAL_SHA256 = "4247ab9ff6398fbb1b6af223629d004265e27bb6cbccabf53ec4969a96c61cab"

TRAINER_DEFAULTS = {
    "--block-dropout": "0.0",
    "--time-reverse-probability": "0.0",
    "--contrast-strength": "0.0",
    "--early-stopping-patience": "0",
    "--weight-decay": "0.0001",
    "--lr": "0.001",
}

COMPONENTS: dict[str, dict[str, Any]] = {
    "separable_headsep_e100_seed42": {
        "short_name": "headsep",
        "weight": 0.25,
        "metadata": ROOT
        / "investigation"
        / "experiments"
        / "parallel100_20260702_separable_headsep_e100_seed42"
        / "small_logmel_cnn_metadata.json",
        "submission": ROOT
        / "investigation"
        / "submissions"
        / "parallel100_20260702_separable_headsep_e100_seed42"
        / "small_logmel_cnn.csv",
        "expected": {
            "n_mels": 128,
            "frames": 512,
            "cache_tag": None,
            "architecture": "separable_residual",
            "activation": "relu",
            "initializer": "he_normal",
            "head_hidden": 256,
            "head_dropout": 0.3,
            "block_dropout": 0.0,
            "optimizer": "adam",
            "effective_weight_decay": 0.0,
            "scheduler": "multistep",
            "lr_milestones": [27, 36, 43, 49, 52],
            "epochs": 100,
            "batch_size": 24,
            "seed": 42,
            "full_train": True,
            "early_stopping_patience": 0,
            "time_reverse_probability": 0.0,
            "contrast_strength": 0.0,
        },
    },
    "globalmel_sep_temporal_e100_seed42": {
        "short_name": "globalmel",
        "weight": 0.375,
        "metadata": ROOT
        / "investigation"
        / "experiments"
        / "parallel100_20260702_globalmel_sep_temporal_e100_seed42"
        / "small_logmel_cnn_metadata.json",
        "submission": ROOT
        / "investigation"
        / "submissions"
        / "parallel100_20260702_globalmel_sep_temporal_e100_seed42"
        / "small_logmel_cnn.csv",
        "expected": {
            "n_mels": 128,
            "frames": 512,
            "cache_tag": "globalmel",
            "architecture": "separable_temporal_bigru",
            "activation": "silu",
            "initializer": "he_normal",
            "head_hidden": 0,
            "head_dropout": 0.3,
            "block_dropout": 0.0,
            "optimizer": "adamw",
            "effective_weight_decay": 1e-4,
            "scheduler": "multistep",
            "lr_milestones": [25, 39],
            "epochs": 100,
            "batch_size": 24,
            "seed": 42,
            "full_train": True,
            "early_stopping_patience": 0,
            "time_reverse_probability": 0.0,
            "contrast_strength": 0.0,
        },
    },
    "sep_temporal_f1024_e100_seed42": {
        "short_name": "f1024",
        "weight": 0.375,
        "metadata": ROOT
        / "investigation"
        / "experiments"
        / "parallel100_20260702_sep_temporal_f1024_e100_seed42"
        / "small_logmel_cnn_metadata.json",
        "submission": ROOT
        / "investigation"
        / "submissions"
        / "parallel100_20260702_sep_temporal_f1024_e100_seed42"
        / "small_logmel_cnn.csv",
        "expected": {
            "n_mels": 128,
            "frames": 1024,
            "cache_tag": None,
            "architecture": "separable_temporal_bigru",
            "activation": "silu",
            "initializer": "he_normal",
            "head_hidden": 0,
            "head_dropout": 0.3,
            "block_dropout": 0.0,
            "optimizer": "adamw",
            "effective_weight_decay": 1e-4,
            "scheduler": "multistep",
            "lr_milestones": [19, 25],
            "epochs": 100,
            "batch_size": 12,
            "seed": 42,
            "full_train": True,
            "early_stopping_patience": 0,
            "time_reverse_probability": 0.0,
            "contrast_strength": 0.0,
        },
    },
}


def sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def notebook_code_sources(path: Path) -> list[str]:
    notebook = json.loads(path.read_text(encoding="utf-8"))
    sources = []
    for cell in notebook["cells"]:
        if cell.get("cell_type") == "code":
            source = cell.get("source", "")
            sources.append("".join(source) if isinstance(source, list) else source)
    return sources


def command_tokens(node: ast.List, source: str) -> list[str]:
    tokens = []
    for element in node.elts:
        if isinstance(element, ast.Constant) and isinstance(element.value, str):
            tokens.append(element.value)
        else:
            tokens.append(ast.get_source_segment(source, element) or "<expr>")
    return tokens


def extract_training_commands(path: Path) -> dict[str, list[str]]:
    for source in notebook_code_sources(path):
        tree = ast.parse(source)
        for statement in tree.body:
            if not isinstance(statement, ast.Assign):
                continue
            if not any(isinstance(target, ast.Name) and target.id == "training_commands" for target in statement.targets):
                continue
            if not isinstance(statement.value, ast.Dict):
                raise ValueError("training_commands is not a dict")
            commands: dict[str, list[str]] = {}
            for key, value in zip(statement.value.keys, statement.value.values):
                if not isinstance(key, ast.Constant) or not isinstance(key.value, str):
                    raise ValueError("training_commands keys must be string literals")
                if not isinstance(value, ast.List):
                    raise ValueError(f"training command for {key.value} is not a list")
                commands[key.value] = command_tokens(value, source)
            return commands
    raise ValueError(f"training_commands not found in {path}")


def flag_value(tokens: list[str], flag: str, default: str | None = None) -> str | None:
    if flag not in tokens:
        return default
    index = tokens.index(flag)
    if index + 1 >= len(tokens):
        raise ValueError(f"{flag} is missing a value")
    return tokens[index + 1]


def command_config(tokens: list[str]) -> dict[str, Any]:
    optimizer = flag_value(tokens, "--optimizer", "adamw")
    weight_decay = float(flag_value(tokens, "--weight-decay", TRAINER_DEFAULTS["--weight-decay"]))
    return {
        "n_mels": int(flag_value(tokens, "--n-mels", "128") or 128),
        "frames": int(flag_value(tokens, "--frames", "512") or 512),
        "cache_tag": flag_value(tokens, "--cache-tag"),
        "architecture": flag_value(tokens, "--architecture", "standard"),
        "activation": flag_value(tokens, "--activation", "silu"),
        "initializer": flag_value(tokens, "--initializer", "default"),
        "head_hidden": int(flag_value(tokens, "--head-hidden", "0") or 0),
        "head_dropout": float(flag_value(tokens, "--head-dropout", "0.35") or 0.35),
        "block_dropout": float(flag_value(tokens, "--block-dropout", TRAINER_DEFAULTS["--block-dropout"]) or 0.0),
        "optimizer": optimizer,
        "effective_weight_decay": weight_decay if optimizer == "adamw" else 0.0,
        "scheduler": flag_value(tokens, "--scheduler", "cosine"),
        "lr_milestones": [
            int(item)
            for item in (flag_value(tokens, "--lr-milestones", "") or "").split(",")
            if item
        ],
        "epochs": int(flag_value(tokens, "--epochs", "12") or 12),
        "batch_size": int(flag_value(tokens, "--batch-size", "24") or 24),
        "seed": int(flag_value(tokens, "--seed", "42") or 42),
        "full_train": "--full-train" in tokens,
        "early_stopping_patience": int(
            flag_value(tokens, "--early-stopping-patience", TRAINER_DEFAULTS["--early-stopping-patience"]) or 0
        ),
        "time_reverse_probability": float(
            flag_value(tokens, "--time-reverse-probability", TRAINER_DEFAULTS["--time-reverse-probability"]) or 0.0
        ),
        "contrast_strength": float(flag_value(tokens, "--contrast-strength", TRAINER_DEFAULTS["--contrast-strength"]) or 0.0),
    }


def metadata_config(path: Path) -> dict[str, Any]:
    raw = json.loads(path.read_text(encoding="utf-8"))
    config = {
        "n_mels": raw["n_mels"],
        "frames": raw["frames"],
        "cache_tag": raw["cache_tag"],
        "architecture": raw["architecture"],
        "activation": raw["activation"],
        "initializer": raw["initializer"],
        "head_hidden": raw["head_hidden"],
        "head_dropout": raw["head_dropout"],
        "block_dropout": raw["block_dropout"],
        "optimizer": raw["optimizer"],
        "scheduler": raw["scheduler"],
        "lr_milestones": raw["lr_milestones"],
        "epochs": raw["best_epoch"],
        "seed": raw["seed"],
        "full_train": raw["full_train"],
        "early_stopping_patience": raw["early_stopping_patience"],
        "time_reverse_probability": raw["time_reverse_probability"],
        "contrast_strength": raw["contrast_strength"],
    }
    weight_decay = float(TRAINER_DEFAULTS["--weight-decay"])
    config["effective_weight_decay"] = weight_decay if raw["optimizer"] == "adamw" else 0.0
    return config


def config_matches(actual: dict[str, Any], expected: dict[str, Any]) -> dict[str, bool]:
    return {
        key: bool(np.allclose(actual[key], value) if isinstance(value, float) else actual[key] == value)
        for key, value in expected.items()
        if key in actual
    }


def validate_blend() -> dict[str, Any]:
    final = pd.read_csv(FINAL_SOURCE)
    label_columns = list(final.columns[1:])
    blended = None
    component_hashes = {}

    for component in COMPONENTS.values():
        path = component["submission"]
        df = pd.read_csv(path)
        component_hashes[component["short_name"]] = sha256(path)
        if list(df.columns) != list(final.columns):
            raise ValueError(f"column mismatch in {path}")
        values = df[label_columns].to_numpy(dtype=float) * float(component["weight"])
        blended = values if blended is None else blended + values

    final_values = final[label_columns].to_numpy(dtype=float)
    max_abs_diff = float(np.max(np.abs(blended - final_values)))
    return {
        "source_csv": str(FINAL_SOURCE.relative_to(ROOT)),
        "source_sha256": sha256(FINAL_SOURCE),
        "source_sha256_matches_expected": sha256(FINAL_SOURCE) == EXPECTED_FINAL_SHA256,
        "component_hashes": component_hashes,
        "max_abs_diff_recomputed_blend": max_abs_diff,
        "recomputed_blend_matches": max_abs_diff < 1e-12,
    }


def main() -> int:
    commands = extract_training_commands(PIPELINE_NOTEBOOK)
    validation: dict[str, Any] = {
        "pipeline": str(PIPELINE_NOTEBOOK.relative_to(ROOT)),
        "components": {},
        "blend": validate_blend(),
    }

    all_ok = True
    for name, component in COMPONENTS.items():
        expected = component["expected"]
        command_actual = command_config(commands[name])
        metadata_actual = metadata_config(component["metadata"])
        command_matches = config_matches(command_actual, expected)
        metadata_matches = config_matches(metadata_actual, expected)
        component_ok = all(command_matches.values()) and all(metadata_matches.values())
        all_ok = all_ok and component_ok
        validation["components"][name] = {
            "weight": component["weight"],
            "metadata": str(component["metadata"].relative_to(ROOT)),
            "submission": str(component["submission"].relative_to(ROOT)),
            "command_config": command_actual,
            "metadata_config": metadata_actual,
            "command_matches_expected": command_matches,
            "metadata_matches_expected": metadata_matches,
            "ok": component_ok,
        }

    all_ok = all_ok and validation["blend"]["source_sha256_matches_expected"]
    all_ok = all_ok and validation["blend"]["recomputed_blend_matches"]
    print(json.dumps(validation, indent=2))
    if all_ok:
        print("final_config_validation_ok")
        return 0
    print("final_config_validation_failed")
    return 1


if __name__ == "__main__":
    raise SystemExit(main())
