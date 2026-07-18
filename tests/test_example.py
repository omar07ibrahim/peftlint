from __future__ import annotations

import subprocess
import sys
from pathlib import Path


def test_verified_inventory_example_runs_from_a_source_checkout() -> None:
    example = Path(__file__).parents[1] / "examples" / "verify_inventory.py"

    completed = subprocess.run(
        [sys.executable, str(example)],
        check=True,
        capture_output=True,
        text=True,
    )

    assert completed.stdout.splitlines() == [
        "PL102 pass <inventory>",
        'PL110 pass tensor:"base_model.model.q_proj.lora_A.weight"',
        'PL111 pass tensor:"base_model.model.q_proj.lora_A.weight"',
        'PL112 pass tensor:"base_model.model.q_proj.lora_A.weight"',
    ]
    assert completed.stderr == ""
