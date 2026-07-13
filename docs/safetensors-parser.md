# Safetensors manifest inspection

`peftlint.safetensors` validates the structural evidence available in a
safetensors v0.8 header. The implementation is pure: it accepts byte values and
declared sizes, performs no file or network I/O, never maps the tensor payload,
and never imports model code.

The format boundary is pinned to safetensors v0.8.0 at commit
[`a406ca3e`](https://github.com/huggingface/safetensors/tree/a406ca3e7a90598be0cd05a50069cb9bf5ef6ba6).
The relevant upstream references are the
[format description](https://github.com/huggingface/safetensors/blob/a406ca3e7a90598be0cd05a50069cb9bf5ef6ba6/README.md#format)
and the
[Rust metadata parser](https://github.com/huggingface/safetensors/blob/a406ca3e7a90598be0cd05a50069cb9bf5ef6ba6/safetensors/src/tensor.rs).

## Pipeline

The explicit API has four stages:

1. `plan_header_read(prefix, file_size=..., limits=...)` validates the exact
   eight-byte prefix and returns the only acceptable header range.
2. `accept_header(plan, header)` requires the planned byte count.
3. `decode_header(envelope)` performs bounded UTF-8, JSON, metadata, and tensor
   schema decoding.
4. `validate_storage(decoded)` proves dtype, shape arithmetic, byte-span,
   layout, and complete payload-coverage invariants.

`parse_safetensors_manifest(prefix, header, file_size=..., limits=...)` composes
those stages when the caller already has both reads. The result is an immutable
`SafetensorsManifest`; tensors use deterministic storage order rather than JSON
member order.

Keeping the stages separate matters for range-capable sources. A source adapter
can inspect the prefix, reject an unreasonable read before requesting it, then
fetch precisely the planned header range.

## Bounded decoding

The default `SafetensorsLimits` are:

| Boundary | Default |
| --- | ---: |
| Header bytes | 16 MiB |
| JSON nesting depth | 32 |
| JSON string characters | 1,048,576 |
| JSON tokens | 250,000 |
| Tensors | 10,000 |
| Tensor rank | 32 |
| Tensor name bytes | 4,096 |
| Metadata entries | 10,000 |

The upstream format limit of 100,000,000 header bytes remains an independent
hard ceiling. Local policy can be stricter but cannot increase it.

Before materializing JSON, the decoder validates UTF-8 and budgets nesting,
tokens, and string length. JSON object keys must be unique at every depth.
Tensor names, metadata, and retained extension-field names must contain valid
Unicode scalar values. Unknown tensor fields are recorded by name and produce a
notice; their values are still syntactically and resource validated but are not
interpreted.

The pinned reader also accepts `__metadata__: null`. That form is preserved as
`MetadataForm.NULL` and surfaced as `HeaderNotice.METADATA_NULL`; other
non-object metadata and non-string metadata values are invalid.

## Dtypes and checked sizes

The accepted tokens are exactly the v0.8 vocabulary:

| Bits per element | Dtypes |
| ---: | --- |
| 4 | `F4` |
| 6 | `F6_E2M3`, `F6_E3M2` |
| 8 | `BOOL`, `U8`, `I8`, `F8_E5M2`, `F8_E4M3`, `F8_E8M0`, `F8_E4M3FNUZ`, `F8_E5M2FNUZ` |
| 16 | `I16`, `U16`, `F16`, `BF16` |
| 32 | `I32`, `U32`, `F32` |
| 64 | `C64`, `F64`, `I64`, `U64` |

The upstream Rust reader represents shape dimensions and offsets as `usize`.
`peftlint` deliberately fixes its supported reader profile at unsigned 64-bit
values so results do not depend on the host running the audit. Element count is
the ordered, checked product of the shape, starting at one. Consequently a
scalar shape `[]` has one element, and zero dimensions affect overflow according
to their position: `[0, 2**63, 2]` is zero-sized, while `[2**63, 2, 0]`
overflows before reaching the zero.

The element count is then multiplied by the dtype width with another unsigned
64-bit check. The resulting bit count must be divisible by eight. This gives
the v0.8 sub-byte rules directly: `F4` needs a multiple of two elements and an
`F6_*` dtype needs a multiple of four. Zero elements are byte-aligned; a scalar
sub-byte tensor is not.

## Layout proof

Tensor spans are sorted by `(begin, end, name)` and checked against a cursor
starting at byte zero. For each span, validation rejects a gap, overlap, reversed
offsets, arithmetic overflow, sub-byte misalignment, or disagreement between
the declared span and the computed tensor size. The final cursor must equal the
payload size derived from the declared file size.

This ordering permits multiple empty tensors at one boundary and sorts them
before a non-empty tensor beginning at the same byte. It does not invent a
natural-alignment rule: for example, a 64-bit tensor may begin immediately after
an 8-bit tensor at byte one when the spans otherwise agree with the format.

## Failure classes

Failures are machine-readable through `SafetensorsErrorCode` and separated by
meaning:

- `SafetensorsReadMismatch` means supplied bytes do not match an exact planned
  read. It does not claim that the underlying artifact is malformed.
- `SafetensorsLimitExceeded` means local inspection policy stopped work before
  a format conclusion could be justified.
- `InvalidSafetensors` means the supplied evidence contradicts the pinned
  format. Its `rule_id` is `PL100` for envelope/schema failures and `PL101` for
  tensor arithmetic or layout failures.

Exception messages and object representations omit tensor names, metadata, raw
header bytes, and extension values so diagnostics do not accidentally disclose
artifact content.

## Source-adapter contract

The pure API cannot prove provenance between two independent byte values. A
filesystem, object-store, or HTTP adapter must therefore:

- read exactly eight prefix bytes, then only the range returned by
  `plan_header_read`;
- bind the prefix, file size, and header read to one unchanged object, using a
  stable file identity, immutable revision, generation, ETag, or equivalent;
- distinguish transport truncation and source mutation from malformed-format
  evidence;
- avoid fetching the tensor payload when only manifest evidence is required.

Passing unrelated prefix and header bytes to the convenience API violates this
contract even if both values happen to satisfy their local checks.

## Non-goals of this component

Manifest inspection does not:

- read files, perform HTTP requests, or validate remote revisions;
- hash the artifact or prove the payload bytes came from the declared object;
- decode tensor values, check numerical contents, or allocate model tensors;
- classify PEFT tensor roles or reconcile LoRA pairs and dimensions;
- compare the adapter against a base-model topology;
- import `transformers`, `peft`, repository code, or `trust_remote_code`.

Those are separate evidence and policy stages. A structurally valid manifest is
necessary input to compatibility analysis, not a claim that an adapter will
load or behave correctly.
