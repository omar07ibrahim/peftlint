# peftlint

Static compatibility evidence for LoRA adapters, without importing model code
or allocating model tensors.

A PEFT adapter is small enough to move casually, but it is not self-contained.
It depends on a particular base model topology, vocabulary, set of target
modules, and often an unspecified revision. A mismatch may remain invisible
until a large base model has been loaded.

`peftlint` moves those failures into a bounded preflight step. It currently
ships pure parsers for PEFT 0.19.1 `adapter_config.json` documents and
safetensors v0.8 manifests, plus a header-only LoRA tensor inventory and four
structural rule evaluators. The broader scanner will reconcile that evidence
with a pinned base-model manifest before inference or deployment.

## What works today

### Adapter configuration

The configuration parser accepts bytes, enforces explicit UTF-8 and JSON
budgets, rejects duplicate keys, expands pinned PEFT defaults, and classifies
all 39 recognized base and LoRA fields without importing PEFT.

```python
from peftlint import parse_adapter_config

manifest = parse_adapter_config(b'{"peft_type":"LORA","r":16,"target_modules":["q_proj","v_proj"]}')

assert manifest.closed_profile
assert manifest.lora is not None
assert manifest.lora.r == 16
```

`closed_profile` means only that the document fits the modeled ordinary-LoRA
configuration schema. It is not a load, safety, or compatibility verdict. See
[Adapter configuration inspection](https://github.com/omar07ibrahim/peftlint/blob/main/docs/adapter-config-parser.md)
for limits, normalization rules, failure classes, and non-goals.

### Safetensors manifest

The safetensors parser exposes a pure four-stage inspection pipeline. It plans
a bounded header read, accepts only the planned byte count, decodes the header
under explicit JSON limits, and proves dtype, shape, span, and payload-coverage
invariants. It never opens the tensor payload or allocates tensor storage.

```python
import json

from peftlint import parse_safetensors_manifest

header = json.dumps(
    {
        "adapter.weight": {
            "dtype": "F32",
            "shape": [2],
            "data_offsets": [0, 8],
        }
    },
    separators=(",", ":"),
).encode("utf-8")

manifest = parse_safetensors_manifest(
    len(header).to_bytes(8, "little"),
    header,
    file_size=8 + len(header) + 8,
)

assert manifest.tensors[0].nbytes == 8
```

The convenience API accepts bytes that a caller has already acquired; it does
not open local paths, fetch URLs, or attest that separate reads came from the
same object. See
[Safetensors manifest inspection](https://github.com/omar07ibrahim/peftlint/blob/main/docs/safetensors-parser.md)
for the staged API, limits, arithmetic, source-adapter contract, and non-goals.

### LoRA inventory evidence

The inventory stage classifies exact PEFT 0.19.1 saved-state keys and evaluates
PL102, PL110, PL111, and a bounded form of PL112. It works only from validated
configuration and safetensors manifest values: no tensor payload, model import,
or user-controlled regular expression is executed. Given already parsed
`config_manifest` and `weights_manifest` values:

```python
from peftlint import evaluate_lora_inventory, inspect_lora_inventory

inventory = inspect_lora_inventory(weights_manifest)
findings = evaluate_lora_inventory(
    config_manifest,
    inventory,
    audit_id=f"audit:sha256:{'0' * 64}",
    artifact="adapter_model.safetensors@example",
)

assert {finding.rule_id for finding in findings} == {
    "PL102",
    "PL110",
    "PL111",
    "PL112",
}
```

The exact `.lora_A.weight` and `.lora_B.weight` suffixes identify a
weight-backed pair, not necessarily a linear layer. Two-dimensional dense
candidates are modeled; higher-rank Conv1d/2d/3d candidates fail closed as
`unknown` until their layer orientation can be reconciled with base topology.
See [Verified LoRA tensor inventory](docs/lora-inventory.md) for the evidence
grammar, rule outcomes, runnable example, and deliberate limits.

## First supported slice

The first ruleset is deliberately narrow:

- Hugging Face PEFT LoRA checkpoints interpreted under PEFT 0.19.1;
- `adapter_config.json` plus `adapter_model.safetensors`;
- modeled two-dimensional weight pairs and embedding targets;
- immutable base-model revisions;
- separate verdicts for loading and adapter hotswapping.

Each verdict has three possible states:

- **compatible** — every required rule was evaluated and passed;
- **incompatible** — a concrete structural contradiction was found;
- **unknown** — the artifact needs runtime validation that static evidence
  cannot justify.

The ruleset requires custom model code, unfamiliar tensor naming, convolution
orientations, fused projections, unsupported configuration features, and
unclassified tensors to produce `unknown`, never a convenient false pass.

End-to-end LoRA compatibility evaluation is still under active development.
The current inventory evaluators produce rule-level structural evidence, not a
standalone load verdict: base-target, selection, dtype, vocabulary, and
remaining mandatory-rule evidence is still absent. The
[LoRA v1 ruleset](https://github.com/omar07ibrahim/peftlint/blob/main/docs/ruleset-v1.md)
defines the compatibility boundary and evidence requirements that the remaining
configuration, base-model, and report stages must satisfy.

## Why static inspection is possible

Safetensors places tensor names, shapes, dtypes, and byte offsets in a bounded
JSON header. For remote artifacts, that header can be fetched with HTTP range
requests rather than downloading the tensor payload. A base-model index locates
the relevant shards; their bounded headers provide the shapes and dtypes needed
for dimension checks. The index alone is not treated as shape evidence.

The core scanner will not import `transformers`, `peft`, a model repository, or
`trust_remote_code`. Runtime execution remains an explicit later validation
step when the static result is `unknown`.

## References

- [PEFT 0.19.1 release](https://github.com/huggingface/peft/releases/tag/v0.19.1)
- [PEFT 0.19.1 checkpoint format](https://github.com/huggingface/peft/blob/v0.19.1/docs/source/developer_guides/checkpoint.md)
- [PEFT 0.19.1 hotswapping](https://github.com/huggingface/peft/blob/v0.19.1/docs/source/package_reference/hotswap.md)
- [Safetensors metadata parsing](https://huggingface.co/docs/safetensors/metadata_parsing)
- [Hugging Face file metadata](https://huggingface.co/docs/huggingface_hub/package_reference/file_download)

## License

Apache-2.0.
