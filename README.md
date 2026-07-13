# peftlint

Static compatibility checks for LoRA adapters, without importing model code
or allocating model tensors.

A PEFT adapter is small enough to move casually, but it is not self-contained.
It depends on a particular base model topology, vocabulary, set of target
modules, and often an unspecified revision. A mismatch may remain invisible
until a large base model has been loaded.

`peftlint` is being built to move those failures into a bounded preflight step.
It will reconcile the adapter configuration, safetensors manifest, and a pinned
base-model manifest before inference or deployment begins.

## First supported slice

The first ruleset is deliberately narrow:

- Hugging Face PEFT LoRA checkpoints interpreted under PEFT 0.19.1;
- `adapter_config.json` plus `adapter_model.safetensors`;
- standard linear and embedding targets;
- immutable base-model revisions;
- separate verdicts for loading and adapter hotswapping.

Each verdict has three possible states:

- **compatible** — every required rule was evaluated and passed;
- **incompatible** — a concrete structural contradiction was found;
- **unknown** — the artifact needs runtime validation that static evidence
  cannot justify.

Custom model code, unfamiliar tensor naming, fused projections, unsupported
configuration features, and unclassified tensors produce `unknown`, never a
convenient false pass.

The executable scanner is under active development. The current repository
defines the compatibility boundary and evidence requirements in
[the LoRA v1 ruleset](docs/ruleset-v1.md).

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
