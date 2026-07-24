# Verified LoRA tensor inventory

`peftlint.lora_inventory` connects a validated safetensors manifest to the
saved-state conventions of PEFT 0.19.1. It classifies exact LoRA keys, compiles
canonical A/B pairs, and evaluates the structural rules PL102, PL110, PL111,
and the bounded part of PL112.

The stage is header-only. It accepts immutable values, performs no file or
network I/O, never reads tensor payloads, and imports neither PEFT nor Torch.
The implementation is pinned to PEFT 0.19.1 commit
[`ba6a190`](https://github.com/huggingface/peft/tree/ba6a19060d6ab54a87538a6e77e3e4d5a907375b).
Its upstream evidence is the pinned
[`get_peft_model_state_dict`](https://github.com/huggingface/peft/blob/v0.19.1/src/peft/utils/save_and_load.py),
[`LoraLayer`](https://github.com/huggingface/peft/blob/v0.19.1/src/peft/tuners/lora/layer.py),
and
[`AuxiliaryTrainingWrapper`](https://github.com/huggingface/peft/blob/v0.19.1/src/peft/utils/other.py)
behavior.

## API

Inspection and evaluation are separate pure stages:

```python
from peftlint import evaluate_lora_inventory, inspect_lora_inventory

inventory = inspect_lora_inventory(weights_manifest)
results = evaluate_lora_inventory(
    config_manifest,
    inventory,
    audit_id="audit:sha256:<64 lowercase hex characters>",
    artifact="adapter_model.safetensors@sha256:<digest>",
)
```

`inspect_lora_inventory` defensively rebuilds and retains the input
`SafetensorsManifest`. Its semantic projection owns the tensor names, dtypes,
shapes, and extension-field names needed by later rules. It retains no path,
URL, source handle, header bytes, or payload bytes.

`evaluate_lora_inventory` defensively rebuilds both manifests and returns a
canonically ordered tuple of `RuleResult` values. It does not produce a full
load verdict: every other mandatory rule in the
[v1 ruleset](ruleset-v1.md) still needs evidence before `summarize_load` can
return `compatible`.

The runnable [header-only example](../examples/verify_inventory.py) constructs
one two-dimensional weight-backed pair and emits four passing structural
findings without allocating the declared payload.

## Exact saved-key grammar

Only four terminal forms are recognized:

| Exact suffix | Saved-key role | Bounded interpretation |
| --- | --- | --- |
| `.lora_A.weight` | weight-backed A | Dense candidate: `[rank, input_features]`; higher-rank Conv candidate: `unknown` |
| `.lora_B.weight` | weight-backed B | Dense candidate: `[output_features, rank]`; higher-rank Conv candidate: `unknown` |
| `.lora_embedding_A` | embedding A | `[rank, vocabulary_size]` |
| `.lora_embedding_B` | embedding B | `[embedding_dim, rank]` |

The first two suffixes identify a saved-key family, not a Linear layer. PEFT
0.19.1 uses them for dense layers and for Conv1d, Conv2d, and Conv3d LoRA
weights. A two-dimensional pair fits the modeled dense orientation. If either
member has more than two dimensions—including the three-, four-, and
five-dimensional PEFT convolution cases—the inventory retains the pair, records
`unmodeled_weight_orientation`, and refuses to reinterpret it without
base-layer evidence.

The target is the complete prefix before that suffix. It must contain nonempty
dot-separated components. A component exactly equal to `lora_A`, `lora_B`,
`lora_embedding_A`, or `lora_embedding_B` makes the key ambiguous and leaves
it unclassified. Prefixes are otherwise preserved exactly: there is no case
folding, Unicode normalization, adapter-name removal, or model-prefix
stripping.

Consequently, internal keys such as `.lora_A.default.weight`, missing or extra
`.weight` components, DoRA magnitude vectors, biases, ordinary saved modules,
and near matches remain unclassified. PEFT removes an adapter name only in its
documented terminal position while saving; the inventory does not attempt a
second normalization pass.

Raw safetensors names may be empty or contain control characters. Evidence
paths therefore use an injective printable namespace: `tensor:` followed by
the JSON string encoding of the complete key. Mixed target-level evidence uses
the separate `target:` namespace. These encodings avoid blank paths and
collisions without changing the underlying names.

## Canonical relationships

Members are ordered by raw key. Pairing is exact on `(target, pair kind)`:
weight-backed and embedding groups at the same target retain their independent
evidence and also produce a mixed-kind closure issue. A complete group owns one
A and one B member. An incomplete recognized group produces an orphan issue;
unclassified tensors never become artificial orphans.

Tensor-record extension fields are retained by the safetensors parser but are
not interpreted here. A canonical-looking member with any such field remains
visible in a syntactic pair and blocks semantic closure with `unknown`.

All public values are frozen and slotted, reject pickle-state restoration, hide
artifact-controlled strings from their default representations, and
reconstruct nested evidence at every public aggregate boundary. Sorting and
uniqueness invariants are checked rather than repaired silently.

## Implemented rules

| Rule | Passing evidence | Fail-closed outcome |
| --- | --- | --- |
| PL102 inventory closure | Every tensor has one modeled role and the config is closed | Empty/unclassified state, extension fields, mixed kinds, higher-rank weight orientations, unavailable config, or nonempty `modules_to_save` is `unknown` |
| PL110 pair completeness | Under a closed config, each recognized group has exactly A and B | A proved orphan is a contradiction; extension fields or a non-closed config make the affected claim `unknown` |
| PL111 pair dimensions | Under a closed config, both modeled shapes are rank two, every dimension is positive, and `A.shape[0] == B.shape[1]` | Without extension fields, a malformed dense/embedding pair is a contradiction; an incomplete, extension-bearing, non-closed, or higher-rank weight case is `unknown` |
| PL112 configured rank | A PL111-valid observed rank equals config `r` when `rank_pattern` is empty | A mismatch is a contradiction; malformed or higher-rank pairs, a non-closed config, or any nonempty `rank_pattern` are `unknown` |

PL112 never compiles or matches `rank_pattern`. PEFT patterns are
user-controlled Python regular expressions with insertion-order semantics;
executing them would break the bounded inspection contract. This implementation
retains their bounded count and returns `unknown` until an equivalent safe
matcher is specified.

With a closed ordinary-LoRA configuration, an empty or entirely unclassified
checkpoint makes PL102 unknown while PL110, PL111, and PL112 pass vacuously:
there are no recognized members to contradict and no pattern to execute. With a
non-closed configuration, all four rules are `unknown`. PL102 remains the
closure gate, so vacuous findings cannot make the complete profile compatible.

## Deliberate boundaries

This stage does not prove:

- that any target exists in a base model;
- whether a weight-backed saved-key pair belongs to a dense or convolution layer;
- input, output, embedding, or vocabulary agreement with base tensors;
- adapter dtype suitability or base-model dtype policy;
- that an unclassified tensor belongs to `modules_to_save`;
- selected-target coverage or nonempty adapter state;
- artifact identity across independent source reads; or
- numerical quality, safety, or inference behavior.

Those dependencies belong to PL120–PL122, PL130, source adapters, and runtime
validation. A passing inventory finding is structural evidence for one rule,
not a standalone load-compatibility claim.
