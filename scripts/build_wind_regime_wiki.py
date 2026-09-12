#!/usr/bin/env python3
"""Encode the static wind-regime Wiki once; training never calls an LLM."""

from __future__ import annotations

import argparse
import os
import sys
from pathlib import Path

import numpy as np
import torch
from transformers import AutoModel, AutoTokenizer

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from utils.wind_regime_wiki import load_wind_regime_wiki_spec


def build_parser():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="configs/wind_regime_wiki.json")
    parser.add_argument("--output", default="outputs/wiki/wind_regime_wiki_qwen.npz")
    parser.add_argument("--llm_path", default="Qwen/Qwen2.5-0.5B")
    parser.add_argument("--device", default="auto")
    parser.add_argument("--batch_size", type=int, default=7)
    parser.add_argument("--max_length", type=int, default=256)
    return parser


def main():
    args = build_parser().parse_args()
    spec = load_wind_regime_wiki_spec(args.config)
    device = (
        "cuda" if args.device == "auto" and torch.cuda.is_available()
        else "cpu" if args.device == "auto"
        else args.device
    )
    tokenizer = AutoTokenizer.from_pretrained(args.llm_path, trust_remote_code=True)
    model = AutoModel.from_pretrained(args.llm_path, trust_remote_code=True).to(device)
    model.eval()
    texts = [
        f"Wind turbine operating scene: {scene['title']}. {scene['prompt']}"
        for scene in spec["scenes"]
    ]
    encoded_parts = []
    with torch.no_grad():
        for start in range(0, len(texts), max(1, int(args.batch_size))):
            tokens = tokenizer(
                texts[start : start + args.batch_size],
                padding=True,
                truncation=True,
                max_length=args.max_length,
                return_tensors="pt",
            ).to(device)
            hidden = model(**tokens, return_dict=True).last_hidden_state
            mask = tokens["attention_mask"].unsqueeze(-1).to(hidden.dtype)
            pooled = (hidden * mask).sum(dim=1) / mask.sum(dim=1).clamp_min(1.0)
            pooled = torch.nn.functional.normalize(pooled.float(), dim=-1)
            encoded_parts.append(pooled.cpu().numpy())
    embeddings = np.concatenate(encoded_parts, axis=0).astype(np.float32, copy=False)
    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    temporary = output.with_name(output.name + f".tmp-{os.getpid()}.npz")
    np.savez_compressed(
        temporary,
        embeddings=embeddings,
        scene_ids=np.asarray(spec["scene_ids"]),
        encoder_name=np.asarray(str(args.llm_path)),
        config_sha256=np.asarray(spec["sha256"]),
        factor_reliability=np.asarray(
            spec.get("factor_reliability") or np.ones(len(spec["scene_ids"])),
            dtype=np.float32,
        ),
    )
    os.replace(temporary, output)
    print(
        "[WIKI] Built frozen semantic anchors: "
        f"scenes={len(spec['scene_ids'])} dim={embeddings.shape[1]} "
        f"encoder={args.llm_path} output={output}"
    )


if __name__ == "__main__":
    main()
