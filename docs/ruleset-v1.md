# LoRA compatibility ruleset v1

## 1. Purpose

This document defines the normative boundary for conclusions that `peftlint`
may eventually draw from static artifacts. Implemented structural components
are identified below; rules without their required evaluator remain contract,
not shipped behavior.

A complete v1 evaluator is intended to answer two different questions:

1. Can this LoRA adapter be loaded onto this exact base-model revision under
   the supported PEFT conventions?
2. Can it replace another adapter through PEFT hotswapping without changing
   the prepared model topology?

A positive answer to either question says nothing about model quality,
training data, safety, or task performance.

Ruleset v1 models PEFT 0.19.1. A later or earlier runtime is a different input,
not a version that this contract assumes is close enough. The normative PEFT
reference is commit
[`ba6a190`](https://github.com/huggingface/peft/tree/ba6a19060d6ab54a87538a6e77e3e4d5a907375b),
which the `v0.19.1` tag resolves to.

| Area | Current status |
| --- | --- |
| Config structural evidence | Implemented |
| Safetensors envelope and storage proof | Implemented |
| Rule-result reducer | Implemented |
| Config-to-PL004 evaluator | Not implemented |
| Inventory and base-model reconciliation | Not implemented |
| Hotswap evaluation | Not implemented |
| Canonical report serialization | Not implemented |

## 2. Inputs

An audit identifies artifacts by immutable content wherever the source permits
it. A mutable branch such as `main` may be resolved, but the resulting report
must record the resolved commit and warn that the input locator was mutable.

### Adapter evidence

- `adapter_config.json`;
- exactly one supported safetensors weight file in the first implementation;
- optional model-card metadata, treated as a claim rather than authority;
- repository revision and per-file identifiers when the source is remote.

### Base-model evidence

- an immutable repository revision;
- `config.json`;
- a safetensors index to locate model shards;
- bounded tensor manifests from each shard needed by a rule;
- tokenizer and embedding metadata when vocabulary checks are requested.

An index maps tensor names to files but does not provide shapes or dtypes. A
dimension rule cannot run until the relevant shard headers have been inspected.

### Runtime evidence

A complete hotswap check targets PEFT 0.19.1 and a named `peftlint` runtime
profile. Its input will include a preparation manifest with:

- the reference adapter identity;
- normalized target paths and layer kinds;
- the rank capacity prepared for each target;
- whether the model was compiled;
- the preparation entry point and options used before hotswapping.

An absent, unsupported, or unmodeled runtime or preparation field must make
hotswap compatibility unknown. Every future hotswap report must include the
full runtime identity and preparation-manifest digest.

The scanner must not import repository code. A model requiring custom code is
not rejected solely for that reason, but rules that depend on its runtime
module semantics return `unknown`.

## 3. Trust and resource boundaries

Static analysis is useful only if inspecting an artifact is cheaper and safer
than loading it. Shipped component parsers enforce the in-memory boundaries
that apply to them; future source adapters and evaluators must preserve the
remaining boundaries in this section.

- JSON inputs have explicit byte and nesting limits.
- The safetensors header length is validated before the header is read.
- Duplicate JSON object keys are rejected during bounded decoding, before a
  manifest is accepted.
- Tensor spans are sorted numerically for validation; JSON member order has no
  storage meaning.
- Tensor data offsets must be contiguous, non-overlapping, and within file
  bounds.
- Declared tensor byte spans must agree with dtype and shape arithmetic.
- Remote reads use bounded requests and verify that the observed object did not
  change between metadata resolution and header retrieval.
- Pickle-based weight files are never deserialized. Their presence is reported
  as unsupported serialization.
- Symbolic links and paths outside a local audit root are not followed.

Component parsers raise a classified limit exception when a resource budget is
exhausted. A future evaluator must translate that condition into `unknown` with
a finding; it is not a pass and not automatically evidence that the adapter is
malformed.

## 4. Verdict model

Each compatibility profile has its own verdict:

| Profile | Question |
| --- | --- |
| `load` | Can the adapter tensors be attached to the inspected base topology? |
| `hotswap` | Can the adapter replace a named reference adapter in an already prepared model? |

Verdicts form a strict three-state model:

| Verdict | Meaning |
| --- | --- |
| `compatible` | All mandatory rules for the profile ran and no contradiction or unresolved dependency remains. |
| `incompatible` | At least one rule produced a reproducible structural contradiction. |
| `unknown` | Available evidence cannot justify either of the other conclusions. |

`incompatible` takes precedence over `unknown`. Otherwise, one mandatory
unknown rule makes the profile unknown. Warnings never upgrade a verdict.

## 5. Rule result shape

Every emitted rule result must contain:

- stable rule identifier and ruleset version;
- severity and compatibility profile;
- artifact identity and logical path;
- a small structural witness, never tensor payload data;
- observed and expected values when both are safe to record;
- whether the rule passed, contradicted the contract, or could not run.

Ordering must be deterministic: profile, rule identifier, artifact, logical
path. The report serializer must use the JSON Canonicalization Scheme in
[RFC 8785](https://datatracker.ietf.org/doc/html/rfc8785). Until that serializer
is implemented and tested, `peftlint` does not claim byte-equivalent canonical
reports.

## 6. Supported configuration profile

The first implementation profile is `peft-0.19.1-lora-v1`. It supports
ordinary LoRA tensors for modeled linear and embedding layers. Compatible
results require `bias="none"`, `lora_bias=false`, `use_dora=false`, and no
`target_parameters`, layer replication, aLoRA invocation tokens, trainable-token
indices, Megatron integration, or unmodeled adapter variant.

Fields such as rank, alpha, dropout, task type, target selection, and
`modules_to_save` are allowed only when the rules below fully interpret them.
The current config parser recognizes special initializers and retains their
normalized kind, but every non-default initializer blocks configuration
closure until later rules model it.

A versioned schema must recognize every config key. Any unknown key or
unsupported non-default value makes the affected profile unknown. This gate is
what prevents a newer PEFT option from being silently treated as ordinary LoRA.
The shipped [adapter configuration parser](adapter-config-parser.md) implements
the structural classification for this gate; `closed_profile` is evidence for
a future PL004 evaluator, not a load verdict by itself.

## 7. Mandatory load rules

### PL001 — checkpoint components

The adapter contains a readable `adapter_config.json` and one supported
safetensors weight file. Missing components are incompatible. Multiple weight
files are unknown until sharded adapters are supported.

### PL002 — adapter method

`peft_type` identifies LoRA. A different PEFT method is unknown under ruleset
v1, not malformed by definition.

### PL003 — unsafe serialization

If the only weights use pickle-based serialization, the audit is unknown and
records why no deserialization occurred. An additional unused pickle file is a
warning unless repository policy promotes it to an error.

### PL004 — configuration closure

Every config key is classified by the `peft-0.19.1-lora-v1` schema, and every
topology- or state-affecting value is supported. An unknown key, an unsupported
non-default value, or a missing value whose PEFT 0.19.1 default cannot be
established makes the load verdict unknown.

The current config parser supplies the field classification and normalized
profile. The PL004 evaluator that converts that evidence into a rule result and
load verdict is not yet implemented.

### PL010 — base identity

The audit resolves a base model from an explicit command-line input or
`base_model_name_or_path`. Conflicting repository names are a provenance policy
failure, not proof of different tensors: the load verdict is unknown until
immutable content identity or complete structural evidence resolves the
conflict. A demonstrated topology contradiction is incompatible.

### PL011 — immutable revision

All remote inputs are resolved to commit identifiers. A missing or mutable
declared revision is a provenance warning; failure to record the resolved
commit makes the audit unknown.

### PL100 — safetensors envelope

The file has the safetensors envelope, a bounded valid JSON header, unique
tensor names, unsigned 64-bit shape and offset components, and an exact dtype
token from the pinned safetensors v0.8 vocabulary. A structural format
violation is incompatible. A dtype introduced after that pinned version is not
silently accepted; a future profile must update the vocabulary deliberately.
The 64-bit bound is peftlint's host-independent supported reader profile for
upstream fields represented as Rust `usize`.

### PL101 — tensor storage

Each offset pair is ordered, tensor byte spans agree exactly with dtype and
shape under checked unsigned 64-bit arithmetic, and the spans form a
hole-free, overlap-free layout after numeric sorting by start, end, and name.
The final span must end at the declared payload boundary: both trailing bytes
and a span beyond the file are PL101 failures. Empty tensors follow the format's
ordered-product and byte-alignment semantics rather than an arithmetic shortcut.

### PL102 — tensor inventory closure

Every checkpoint tensor has exactly one understood role: a supported LoRA A/B
member, a supported embedding LoRA member, or a tensor governed by an explicit
`modules_to_save` rule. DoRA magnitude vectors, bias tensors, adapter variants,
and any other unclassified state make the load verdict unknown. A known tensor
that contradicts the config is incompatible.

### PL110 — LoRA pair completeness

Every supported `lora_A` tensor has exactly one corresponding `lora_B` tensor,
and vice versa. Adapter-name prefixes are normalized only through documented
PEFT conventions. Orphans are incompatible. If distinct raw names collapse to
the same normalized target or pair, the result is unknown unless the runtime
profile defines an unambiguous selection.

### PL111 — LoRA pair dimensions

For a standard linear target, `A` has shape `[rank, input_features]` and `B`
has shape `[output_features, rank]`. The inner ranks must agree. For embeddings,
the supported PEFT orientation is checked separately. Unknown orientations do
not pass through the linear rule.

### PL112 — configured rank

Observed pair ranks agree with `r` after replaying PEFT 0.19.1's first-match,
insertion-order suffix-regex lookup for `rank_pattern`. The report warns when
several patterns could match. If source member order was not preserved or the
matching feature cannot be modeled exactly, the result is unknown. A
deterministically selected rank mismatch is incompatible.

### PL120 — base target existence

Each normalized adapter target maps to one base-model tensor group under a
named architecture mapping. A proven missing target is incompatible. Multiple
possible matches are unknown unless the pinned runtime and architecture mapping
define a deterministic selection. If the architecture mapping is unavailable,
the check is unknown rather than inferred from a similar model family.

### PL121 — base target dimensions

The base tensor dimensions agree with the adapter pair's input and output
features. Fused QKV projections, Conv1D orientation, expert parameters, and
layer replication require explicit rules; otherwise this check is unknown.

### PL122 — target-selection closure

The PEFT 0.19.1 semantics of `target_modules`, `exclude_modules`, layer filters,
and the supported target patterns select exactly the checkpoint targets after
normalization. A deterministic selected target missing from the checkpoint, or
an unexplained checkpoint target outside that selection, is incompatible.
Unsupported selectors such as `target_parameters` make the result unknown
through PL004.

### PL130 — modules to save

Tensors declared through `modules_to_save` are separated from LoRA pairs and
must map to the expected full base tensors. Missing, partial, or unexpected
saved modules are incompatible when the mapping is known. Colliding normalized
module paths are unknown unless PEFT 0.19.1 behavior resolves them uniquely.

### PL140 — vocabulary-sensitive tensors

Embedding and language-model-head tensors are checked against the inspected
base vocabulary and tying policy. A vocabulary-size contradiction is
incompatible. Token meaning or tokenizer equivalence is outside a shape-only
check and may remain unknown.

## 8. Mandatory hotswap rules

A future hotswap evaluator must compare a candidate with a reference adapter
and the prepared base topology. Both load verdicts must be compatible. An
incompatible load verdict makes hotswap incompatible; otherwise, an unknown
load verdict makes hotswap unknown before hotswap-specific rules are considered.

### PL200 — method compatibility

The PL200 evaluator must reproduce PEFT 0.19.1's
`check_hotswap_configs_compatible` gate. Both adapters must have the same
`peft_type` and equal values for `use_rslora`, `lora_dropout`, `alpha_pattern`,
and `use_dora`. A known mismatch is incompatible. A missing or unmodeled
comparison value is unknown rather than filled from a different PEFT release.

### PL201 — runtime profile

The requested runtime is PEFT 0.19.1 and the complete preparation manifest
matches a runtime profile modeled by this ruleset. Missing or unmodeled runtime
evidence is unknown. Version-specific behavior is never inferred from a nearby
release.

### PL210 — prepared target coverage

The candidate does not require target layers absent from the prepared reference
topology. A candidate may use fewer prepared layers only when the supported
runtime contract defines how unused layers are zeroed.

### PL211 — rank capacity

Candidate ranks do not exceed the prepared maximum rank for their targets.
Rank padding is described in evidence; it is never silently assumed.

### PL212 — saved-module stability

`modules_to_save`, embedding resize requirements, and other topology-changing
components are compatible with replacement in place. If runtime behavior is
version-dependent and the exact behavior is not modeled, the verdict is
unknown.

## 9. Ruleset limitations

Ruleset v1 does not claim static proof for:

- AdaLoRA, IA3, prompt tuning, or other non-LoRA PEFT methods;
- arbitrary custom model code;
- quantized or packed base tensors whose logical shapes cannot be recovered;
- fused or architecture-specific projections without an explicit mapping;
- numerical correctness of merging or inference;
- semantic tokenizer equivalence;
- hotswap behavior outside a named, tested PEFT runtime profile;
- adapter quality, safety, or licensing.

Support for one architecture does not generalize by naming resemblance. New
architecture mappings and PEFT variants require fixtures that demonstrate both
valid artifacts and minimal counterexamples.

## 10. Evidence required before a non-unknown verdict ships

An evaluator may emit a non-unknown result for a rule only after tests cover:

1. a valid minimal fixture;
2. a single-fault fixture that the rule rejects;
3. boundary values for sizes, ranks, and offsets;
4. deterministic evidence serialization;
5. interaction with every earlier mandatory rule in the same profile.

Until that evidence exists, the rule is unimplemented and the affected profile
must remain `unknown`.
