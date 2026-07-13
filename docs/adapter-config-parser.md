# Adapter configuration inspection

`peftlint.adapter_config` turns one complete `adapter_config.json` byte string
into bounded, immutable structural evidence. It performs no file or network
I/O, imports no model or PEFT code, and never evaluates retained module names,
regular expressions, or custom-class mappings.

The interpretation is pinned to PEFT 0.19.1 at commit
[`ba6a190`](https://github.com/huggingface/peft/tree/ba6a19060d6ab54a87538a6e77e3e4d5a907375b).
The modeled upstream surfaces are the pinned
[`PeftConfig`](https://github.com/huggingface/peft/blob/ba6a19060d6ab54a87538a6e77e3e4d5a907375b/src/peft/config.py),
[`LoraConfig`](https://github.com/huggingface/peft/blob/ba6a19060d6ab54a87538a6e77e3e4d5a907375b/src/peft/tuners/lora/config.py),
and
[`TaskType`](https://github.com/huggingface/peft/blob/ba6a19060d6ab54a87538a6e77e3e4d5a907375b/src/peft/utils/peft_types.py)
definitions. A `peft_version` member is retained as provenance; it does not
switch the parser to another set of semantics.

## API

The public entry point accepts bytes and optional resource limits:

```python
from peftlint import parse_adapter_config

manifest = parse_adapter_config(
    b'{"peft_type":"LORA","r":16,'
    b'"target_modules":["q_proj","v_proj"]}'
)

assert manifest.closed_profile
assert manifest.lora is not None
assert manifest.lora.r == 16
```

`AdapterConfigManifest` records:

- `method_status`, separating supported LoRA, missing or invalid method
  declarations, and other PEFT methods;
- `explicit_fields`, the sorted root member names present in the document;
- an optional normalized `LoraConfigProfile`;
- redacted `issues` that classify fields outside the ordinary profile; and
- `closed_profile`, a convenience property for configuration closure.

Configuration closure is deliberately narrow. It means the document identifies
LoRA, every modeled core field is valid, and no blocking field issue remains.
It does not prove that weights exist, tensor dimensions agree, a base model is
the right one, the adapter is safe, or loading will succeed.

## Resource limits

The default `AdapterConfigLimits` are:

| Boundary | Default |
| --- | ---: |
| Document bytes | 1 MiB |
| JSON nesting depth | 32 |
| JSON string characters | 65,536 |
| JSON tokens | 50,000 |
| JSON number characters | 128 |
| Root fields | 256 |
| Members in one collection | 10,000 |
| Retained name bytes | 4,096 |

The document byte limit is checked before UTF-8 decoding. Bounded JSON decoding
then enforces nesting, string, token, and number budgets while detecting
duplicate object keys. Root-field, collection, retained-name, and Unicode
scalar checks run before a manifest is returned.

Integer and decimal policies are host independent. Integers retained by the
profile fit a signed 64-bit range. Decimal values are preserved exactly, must
be finite and within the modeled field range, and use a fixed exponent bound
rather than the host Python build's `Decimal` limits.

## Field classification

The schema recognizes all 39 pinned base and LoRA root fields.

- Ordinary core fields are normalized into `LoraConfigProfile`.
- An invalid core type, value, or cross-field combination prevents construction
  of that profile and produces a blocking issue.
- Non-default bias modes and every recognized special initializer are retained
  for evidence but remain blocking.
- LoRA variant fields are accepted only at their pinned ordinary defaults;
  valid non-default values are classified as unsupported.
- `runtime_config` is recognized, ignored, and nonblocking. Its opaque value is
  validated only for JSON structure and resource use, then discarded.
- An unknown root field is blocking. Its name is classified, while its opaque
  value is not retained.

An unsupported `peft_type` is not malformed by definition. The manifest records
the method as unsupported and does not apply LoRA field semantics to the rest
of that document.

## Normalization without execution

Missing members use PEFT 0.19.1 defaults. Decimal lexemes remain `Decimal`
values rather than binary floats. Target and exclusion name lists are converted
to deterministic sorted sets; `modules_to_save`, `rank_pattern`, and
`alpha_pattern` preserve the ordering needed by their pinned semantics. Layer
indices are sorted and deduplicated.

The `all-linear` selector is recognized case-insensitively. Other string module
selectors remain opaque regular-expression text. The parser never compiles or
matches those expressions. Likewise, `auto_mapping` strings are validated and
stored without importing the named library or class.

All public evidence models revalidate and defensively rebuild nested values.
Their default representations omit retained user-controlled strings, and
classified exception messages do not echo document content.

## Failure classes

Document failures are separate from schema evidence:

- `InvalidAdapterConfig` reports malformed UTF-8 or JSON, duplicate keys,
  invalid Unicode scalars, and a non-object root.
- `AdapterConfigLimitExceeded` reports which inspection policy stopped work and
  includes the applicable numeric limit.
- A recognized field with a bad type, invalid value, unsupported value, or
  ignored status normally becomes a `ConfigFieldIssue` in a returned manifest.

This separation lets a future evaluator distinguish malformed input, local
inspection policy, an unsupported configuration, and an ordinary closed
profile without turning every unfamiliar field into an exception.

## Non-goals

This component does not:

- open paths, fetch repositories, resolve revisions, or prove source identity;
- import `peft`, `transformers`, model repositories, or custom code;
- execute regular expressions or custom-class mappings;
- inspect adapter weights or reconcile configuration with tensor names;
- resolve a base model or inspect its topology;
- produce `RuleResult` objects or a load or hotswap verdict.

Those steps require additional evidence and remain separate from configuration
parsing under the [LoRA v1 ruleset](ruleset-v1.md).
