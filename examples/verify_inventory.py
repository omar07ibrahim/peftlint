"""Build and evaluate a tiny header-only LoRA checkpoint fixture."""

from __future__ import annotations

import json

from peftlint import (
    evaluate_lora_inventory,
    inspect_lora_inventory,
    parse_adapter_config,
    parse_safetensors_manifest,
)


def main() -> None:
    payload_size = 256
    header = json.dumps(
        {
            "base_model.model.q_proj.lora_A.weight": {
                "dtype": "F32",
                "shape": [4, 7],
                "data_offsets": [0, 112],
            },
            "base_model.model.q_proj.lora_B.weight": {
                "dtype": "F32",
                "shape": [9, 4],
                "data_offsets": [112, payload_size],
            },
        },
        separators=(",", ":"),
    ).encode("utf-8")
    weights = parse_safetensors_manifest(
        len(header).to_bytes(8, "little"),
        header,
        file_size=8 + len(header) + payload_size,
    )
    config = parse_adapter_config(b'{"peft_type":"LORA","r":4,"target_modules":["q_proj"]}')

    inventory = inspect_lora_inventory(weights)
    results = evaluate_lora_inventory(
        config,
        inventory,
        audit_id=f"audit:sha256:{'0' * 64}",
        artifact="adapter_model.safetensors@example",
    )
    for result in results:
        print(result.rule_id, result.outcome.value, result.logical_path or "<inventory>")


if __name__ == "__main__":
    main()
