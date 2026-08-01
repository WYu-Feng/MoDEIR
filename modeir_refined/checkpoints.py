from __future__ import annotations

from pathlib import Path
from typing import Any

import torch
import torch.nn as nn


def load_payload(path: str | Path, *, mmap: bool = False) -> dict[str, Any]:
    path = Path(path)
    if not path.is_file():
        raise FileNotFoundError(path)
    kwargs: dict[str, Any] = {"map_location": "cpu", "weights_only": False}
    if mmap:
        kwargs["mmap"] = True
    try:
        payload = torch.load(path, **kwargs)
    except TypeError:
        kwargs.pop("weights_only", None)
        kwargs.pop("mmap", None)
        payload = torch.load(path, **kwargs)
    if not isinstance(payload, dict):
        raise TypeError(f"Checkpoint must contain a dict: {path}")
    return payload


def load_state(module: nn.Module, state: dict[str, torch.Tensor], label: str, *, strict: bool = False):
    missing, unexpected = module.load_state_dict(state, strict=strict)
    print(f"[LOAD] {label}: missing={len(missing)} unexpected={len(unexpected)}")
    if strict and (missing or unexpected):
        raise RuntimeError(f"Strict state load failed for {label}")
    return list(missing), list(unexpected)


def load_compatible_state(
    module: nn.Module,
    state: dict[str, torch.Tensor],
    label: str,
    *,
    skip_prefixes: tuple[str, ...] = (),
):
    own_state = module.state_dict()
    compatible = {}
    skipped = []
    unexpected = []
    for key, value in state.items():
        if any(key.startswith(prefix) for prefix in skip_prefixes):
            skipped.append(key)
            continue
        if key not in own_state:
            unexpected.append(key)
            continue
        if tuple(own_state[key].shape) != tuple(value.shape):
            skipped.append(key)
            continue
        compatible[key] = value
    missing, load_unexpected = module.load_state_dict(compatible, strict=False)
    unexpected.extend(load_unexpected)
    print(
        f"[LOAD] {label}: compatible={len(compatible)} missing={len(missing)} "
        f"unexpected={len(unexpected)} skipped={len(skipped)}"
    )
    return list(missing), list(unexpected), list(skipped)


def cpu_state(module: nn.Module) -> dict[str, torch.Tensor]:
    return {key: value.detach().cpu() for key, value in module.state_dict().items()}


def decoder_refine_state(decoder: nn.Module) -> dict[str, Any]:
    return {
        "frm_mid32": cpu_state(decoder.frm_mid32),
        "frm_up": [cpu_state(block) for block in decoder.frm_up],
    }


def load_decoder_refine(decoder: nn.Module, payload: dict[str, Any]) -> None:
    load_state(decoder.frm_mid32, payload["frm_mid32"], "FRM mid32", strict=False)
    if len(payload["frm_up"]) != len(decoder.frm_up):
        raise RuntimeError("FRM up-block count mismatch")
    for index, (block, state) in enumerate(zip(decoder.frm_up, payload["frm_up"])):
        load_state(block, state, f"FRM up[{index}]", strict=False)
