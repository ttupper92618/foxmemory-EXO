---
id: model-cards
title: Model Cards
sidebar_position: 4
---

<!-- Copyright 2025 Foxlight Foundation -->

Model cards are Skulk's durable source of truth for model metadata.

They are how Skulk knows things like:

- what a model is called
- what task types it supports
- how large it is for placement and download planning
- whether it supports tensor sharding
- whether it has vision support
- whether it declares advanced model-specific behavior

## What Model Cards Do

Model cards sit at the boundary between static metadata and runtime behavior.

They drive:

- placement and memory calculations
- model browsing in the API and dashboard
- custom model registration
- exact pre-publication qualification without claiming registry trust
- modality hints such as vision
- advanced capability declarations for model-specific runtime behavior

## Where They Live

Skulk's current supported catalog comes from the TUF-verified external registry
at `registry.foxlight.ai`. A registry card represents one exact selectable
artifact—one quant or selected file—and carries an immutable card ID and signed
snapshot provenance. Registry refreshes do not require a Skulk release.

An authenticated operator may also install a complete pinned card through
`POST /models/add-card`. Skulk preserves its artifact bundle but strips every
signed-registry trust claim and stores it as a custom card. This lets the model
registry exercise an exact candidate on real hardware before publication while
keeping publication and model-level repository-code approval as separate trust
decisions.
Headless qualification may use a dedicated
`SKULK_EXACT_CARD_QUALIFICATION_TOKEN` bearer for only the temporary install and
cleanup operations. The token does not approve repository code and does not
grant any wider operator or inference scope. Skulk requires the immutable source
revision and immutable pins for every external companion repository, assigns a
durable `qualification_only` marker only to service-authenticated installs, and
refuses to replace or remove any pre-existing non-qualification card through
this service credential. The elected master rechecks that ownership at its
serialized command-ordering boundary, closing races between different API
nodes. Success additionally requires local persistence of the indexed event
carrying that exact command ID and visibility of the exact card, so a cached
identical card cannot acknowledge a new qualification request.
Cleanup waits for its own indexed delete acknowledgement. Downloaded bytes and
their installed record remain available for later signed adoption, while the
`qualification_only` sidecar is deliberately excluded from catalog projection
once its lifecycle-owned custom card has been removed.
The qualification worker also sends the candidate's `artifact_bundle_id` to
the store download endpoint. Both the API node and canonical store verify that
identity, so a changed alias cannot redirect qualification to different bytes.

The signed catalog also carries open `architecture` and `capability_claims`
metadata beside the immutable card. Claims describe intrinsic model behavior
and selected-artifact completeness without asserting that Skulk can serve the
capability today. A separately signed engine-support matrix records exact
engine-build compatibility; neither discovery agents nor mutable card fields
can turn a claim into placement permission. Empirical load and feature
qualification is bound to the exact immutable card it tested. An explicit
artifact-scoped `incomplete` claim blocks matrix-derived placement for that
capability even when the base model advertises it.

Registry envelope v2 can additionally carry an `artifact_bundle`: a strict,
content-derived description of one complete executable artifact. It pins an
optional repository-relative loader root and the exact required files, sizes,
and upstream object identities. This lets several independently loadable quants
share one Hugging Face repository and revision without collapsing into one
card. Existing v1 cards remain valid and retain their historical download
behavior.

Complete canonical and staged artifacts retain their full effective card and
hashed manifest beside the bytes in `.skulk/installed-card.json`. These
installed cards load before registry access, remain usable indefinitely while
their artifact is complete, and keep an older installed generation active until
a replacement has transferred and verified atomically.

Fallback cards are still shipped in:

- [`resources/inference_model_cards`](https://github.com/Foxlight-Foundation/Skulk/tree/main/resources/inference_model_cards)
- [`resources/image_model_cards`](https://github.com/Foxlight-Foundation/Skulk/tree/main/resources/image_model_cards)
- [`resources/embedding_model_cards`](https://github.com/Foxlight-Foundation/Skulk/tree/main/resources/embedding_model_cards)

They provide a startup catalog when registry access and its bounded verified
cache are unavailable; they do not replace the signed registry as current
catalog truth.

Custom cards are stored under the user data directory and synced through the
cluster event flow. They are operator-owned and retain final precedence over
registry, installed, and bundled cards for the same `model_id`.

## The card interface (source of truth)

The authoritative definition of the model-card interface is the `ModelCard`
type in
[`src/skulk/shared/models/model_cards.py`](https://github.com/Foxlight-Foundation/Skulk/tree/main/src/skulk/shared/models/model_cards.py).
Every field is documented in that model, and the exhaustive, always-current field
reference is the generated API schema (`ModelCard` and its nested
`PlacementCardConfig` / `RuntimeCapabilityCardConfig` / `VisionCardConfig` /
`ReasoningCardConfig` / `ModalitiesCardConfig` / `ToolingCardConfig` /
`ComponentInfo`) in the [API reference](/api/skulk-api). This page is the curated
narrative; when in doubt about an exact field, the schema is canonical.

Cards are camelCase on the wire and strict (unknown fields are rejected), so every
node in a cluster must run the same Skulk version.

## Core Fields

### Identity and size

- `model_id`
  - selectable artifact alias; legacy and bundled cards normally use the Hugging Face repository id, while registry cards may give two exact files or quants from one repository different aliases
- `source_repository`
  - optional upstream Hugging Face repository containing the bytes; defaults to `model_id` and is set by the registry when the selectable alias differs from the byte origin
- `storage_size`
  - total model size used for store/download/placement planning
- `n_layers`
  - number of transformer layers used for pipeline sharding
- `hidden_size`
  - hidden dimension used for placement and compatibility checks
- `num_key_value_heads`
  - optional KV head count for tensor compatibility decisions
- `gguf_file`
  - for GGUF (llama.cpp) models only: the repo-relative weights file the runner loads (the selected quant's first shard), resolved once at card creation; `null` for safetensors/MLX cards
- `source_revision`
  - optional full Hugging Face commit hash for the qualified model artifacts; when set, metadata, store downloads, direct downloads, and worker staging all use that immutable revision instead of the repository's mutable `main` branch
- `artifact_bundle`
  - optional signed v2 manifest for one exact executable artifact: `artifact_root`,
    `files` (repository-relative path, size, and optional immutable upstream
    object identity), `bundle_identity`, `download_size`, and equivalent
    alternate locations
  - signed v2 cards require this manifest to be internally consistent. Paths
    are canonical POSIX-relative paths and cannot escape the repository or the
    declared artifact root
  - the loader runs from `artifact_root`, while file paths such as `gguf_file`
    remain repository-relative for compatibility
- `components`
  - for multi-component models (such as a diffusion stack): the per-component weight layout; `null` for a single-weights model

### Runtime and placement

- `supports_tensor`
  - whether tensor-style placement is allowed (GGUF/llama.cpp cards set this `false`)
- `tasks`
  - supported task families such as `TextGeneration`, `TextEmbedding`, image tasks, `TextToSpeech`, `SpeechToText`, or `SpeechTranslation`
- `trust_remote_code`
  - whether the artifact requires repository-supplied Python; signed publication authorizes the exact immutable registry card regardless of provenance
  - explicitly adding an external model authorizes its pinned card, and an omitted Hugging Face revision is resolved to one immutable commit before the card is created; bundled cards are authorized by the Skulk release that ships them
  - legacy executable custom cards that predate immutable revision pinning fail closed and must be re-added through the operator flow; an absent revision can never silently authorize mutable `main`
  - ordinary catalog reads and placement requests never fetch or persist an unknown Hub card; callers must use an authenticated add flow first, and exact-placement payloads must match that effective catalog card completely
  - this field controls the loader's repository-code behavior, not a second operator approval ceremony; artifact identity and immutable revision checks still fail closed
- `uses_cfg`
  - whether the model uses classifier-free guidance (relevant to some image/diffusion models)

### Catalog metadata

- `registry_card_id` / `registry_snapshot_id` / `registry_provenance`
  - provenance is signed catalog metadata (`foxlight`, `agent`, or `community`)
    and is deliberately excluded from the content-derived card identity
  - runtime provenance attached by the verified external catalog; these are absent from bundled and custom cards
- `registry_architecture` / `registry_capability_claims`
  - signed intrinsic architecture and model/artifact capability evidence from
    the registry envelope; persisted into installed sidecars for air-gapped use
  - these fields do not declare present-day Skulk compatibility. Placement may
    expand only from a separate exact signed engine/build support decision.

- `family`
  - coarse family label such as `gemma`, `qwen`, `deepseek`
- `quantization`
  - human-facing quantization label
- `base_model`
  - display-friendly base model name
- `context_length`
  - advertised context length if known
- `capabilities`
  - coarse capability list such as `text`, `vision`, `thinking`, `embedding`, `tts`, or `stt`

These coarse capabilities remain useful for browsing, badges, and basic compatibility, but they are not expressive enough for model-specific runtime behavior on their own.

## Vision Section

`[vision]` is the existing structured section for multimodal text-generation models.

Fields include:

- `image_token_id`
  - token used to represent image slots in prompt tokenization
- `model_type`
  - MLX-VLM model family identifier such as `gemma4`
- `weights_repo`
  - optional alternate weights repository for the vision tower
- `weights_revision`
  - full immutable commit for a separate `weights_repo`; required for signed-registry cards
- `image_token`
  - optional literal image token string
- `processor_repo`
  - optional alternate processor repository
- `processor_revision`
  - full immutable commit for a separate `processor_repo`; required for
    signed-registry cards because processor loaders may execute repository Python
- `boi_token_id`
  - optional begin-of-image token id
- `eoi_token_id`
  - optional end-of-image token id
- `projector_file`
  - exact repository-relative GGUF multimodal projector selected for served
    vision; when present, `gguf_file` and a full immutable `source_revision`
    are also required
- `projector_size`
  - exact positive byte size of `projector_file` at `source_revision`; it must
    appear together with `projector_file`

Legacy GGUF vision cards without an exact projector pin remain loadable through
the in-process `llama_cpp` path. `llama_server` becomes eligible only when both
projector fields are present, so the served runner never guesses among upstream
projector variants.

## Placement Section

`[placement]` declares where a model is allowed to run and which backend is
preferred. It is what the planner reads to route a model to suitable nodes, and
it is how a heterogeneous cluster keeps each model on hardware that can run it.

Backends are named `<engine>-<compute>` tags (for example `mlx-metal`,
`llama_cpp-vulkan`, `llama_cpp-rocm`, `llama_cpp-cpu`). The engine selects the
worker runner; the compute names the accelerator. A bare engine tag (e.g. `mlx`)
is also valid vocabulary: nodes advertise it alongside their compound tags, so a
card written against the original `{"mlx"}` set keeps matching.

- `compatible_backends`
  - the hard filter: the set of backend tags this model may run on. The planner excludes any node whose advertised backends do not intersect this set. The default is `{"mlx"}` (so an unannotated card stays on MLX nodes); a GGUF card lists the llama.cpp tags. This is what keeps an MLX model off an AMD node and a GGUF model off a Mac.
- `backend_preference`
  - the soft score: an ordered list of preferred tags. When several compatible nodes qualify, the planner prefers the node whose backend ranks earliest, with graceful fallback to the rest. This lets a card say "fastest on Vulkan, but ROCm is fine."
- `min_vram_gib`
  - optional minimum accelerator memory a node must have to be eligible.
- `max_context_tokens`
  - optional cap on the admission context for this model, independent of the model's trained context length.
- `max_pipeline_split_layer`
  - optional largest layer boundary where a later pipeline rank may begin. Use it when a model's tail reuses KV from earlier concrete layers; the planner shifts proportional boundaries left as needed and reruns normal per-node memory checks so the final rank owns every producer it consumes.

## Extended Capability Sections

Skulk now supports optional structured sections that declare refined model behavior.

Existing cards do **not** need these sections. If they are absent, Skulk falls back to generic behavior.

### `[reasoning]`

Declares advanced reasoning behavior:

- `supports_toggle`
  - whether thinking/reasoning can be explicitly enabled or disabled
- `supports_budget`
  - whether the model supports a reasoning budget control
- `format`
  - reasoning marker format such as `channel_delimited` or `token_delimited`
- `default_effort`
  - reasoning effort used when thinking is enabled without an explicit effort
- `disabled_effort`
  - reasoning effort used when thinking is explicitly disabled

### `[modalities]`

Declares refined modality support:

- `supports_audio_input`
  - whether the model supports audio input
- `supports_native_multimodal`
  - whether the model uses a native multimodal path rather than generic text-only prompting

### `[audio]`

Declares speech serving metadata for TTS and STT models:

- `kind`
  - `tts` for text-to-speech or `stt` for speech-to-text
- `default_response_format`
  - default encoded audio format for TTS, such as `mp3`
- `response_formats`
  - encoded audio formats the model can produce, such as `mp3`, `wav`, `flac`, `ogg`, or `opus`
- `supports_streaming`
  - whether the model can stream partial speech or transcription output; keep
    this false until the Skulk runtime has validated the model/backend path
- `supports_realtime`
  - whether the model exposes a realtime audio session interface
- `supports_voice_listing`
  - whether voices can be enumerated by the serving API
- `voices`
  - stable model-native or bundled-reference identifiers returned by
    `GET /v1/audio/voices`; this
    requires `kind = "tts"` and `supports_voice_listing = true`
- `voice_catalog`
  - optional ordered metadata for every identifier in `voices`; each entry
    carries the same `id`, a display `name`, and ordered BCP 47
    `preferred_languages` used by clients for deterministic language matching;
    `reference_profile` names a checksummed profile under
    `resources/speech_reference_voices/` when the voice is reference-conditioned
- `default_voice`
  - stable voice used when a TTS request omits `voice`; it must appear in
    `voices`
- `supports_reference_audio`
  - whether managed reference audio can condition the voice
- `supports_translation`
  - whether speech-to-English translation is supported through the
    standard `/v1/audio/translations` route
- `sample_rates`
  - supported input or output sample rates in hertz

For stable TTS streaming, `audio.supports_streaming = true` is the model-side
eligibility gate; the model must also be mounted and ready.

For realtime STT, both `supports_streaming = true` and
`supports_realtime = true` are necessary but not sufficient. The API must have
reachable ready single-host capacity and use a model whose upstream runtime
exposes a true incremental streaming session. The bundled
`mlx-community/Voxtral-Mini-4B-Realtime-2602-4bit` card is the first validated
contract candidate. Batch Parakeet and Whisper cards deliberately keep both
flags false.

### `[tooling]`

Declares tool-calling behavior:

- `supports_tool_calling`
  - whether tool calling is supported
- `builtin_tools`
  - optional list of builtin platform tool contracts such as `web_search`, `open_url`, or `extract_page`
- `tool_call_format`
  - expected tool-call output format such as `generic`, `gemma4`, `gpt_oss`, `dsml`, or `atem` (Muse Glimmer's `<atem:function_calls>` markup)

### `[runtime]`

Declares runtime integration preferences:

- `prompt_renderer`
  - prompt renderer to use, such as `tokenizer`, `gemma4`, or `dsml`
- `output_parser`
  - output parser to use, such as `generic`, `gemma4`, `gpt_oss`, `deepseek_v32`, or `muse_glimmer` (the `to=self` / `to=user` / `to=<tool>` channel grammar)
- `metal_fast_synch`
  - per-model override for the MLX `MLX_METAL_FAST_SYNCH` flag; set to `false` for models that deadlock under FAST_SYNCH on the ring backend
- `mtp_heads`
  - set to `true` when the model has native MTP prediction heads available via sidecar; required alongside `mtp_sidecar_repo` to enable speculative decoding
- `mtp_max_depth`
  - maximum draft depth the MTP heads support; start at `1` for Apple Silicon (deeper values rarely amortize on Metal due to near-linear verify-pass scaling)
- `mtp_sidecar_repo`
  - Hugging Face repo ID containing the published `mtp.safetensors` sidecar (e.g. `"FoxlightAI/qwen3-5-9b-base-mtp"`); produced by SWP (Skulk Weights Publisher) from the original BF16 checkpoint
- `mtp_sidecar_revision`
  - full immutable commit for a separately hosted MTP sidecar; required for signed-registry cards
- `mtp_norm_convention`
  - how the MTP heads normalize hidden states (`zero_centered` or `actual_scale`); must match how the sidecar was produced
- `mtp_concat_order`
  - the order the MTP heads concatenate the embedding and hidden state (`embed_first` or `hidden_first`); must match the sidecar
- `speculative_multi_node`
  - set to `false` to forbid speculative decoding when the model is sharded across multiple nodes (it stays single-node speculative); the runner and the generation loop read this to make the same rank-symmetric decision
- `assistant_model_repo`
  - Hugging Face repo of a small companion model used as an external drafter for speculative decoding (the Gemma 4 path), as opposed to native MTP heads
- `assistant_model_revision`
  - full immutable commit for a separately hosted assistant; required for signed-registry cards

Separate `served_spec_draft_repo` and `vllm_spec_draft_repo` companions likewise
require `served_spec_draft_revision` and `vllm_spec_draft_revision`. A companion
in the base artifact repository inherits the card's `source_revision`.

## MTP Speculative Decoding

Some models include native multi-token prediction (MTP) heads baked into their checkpoint weights. Skulk uses these heads for speculative decoding: the MTP heads draft candidate tokens cheaply, a full forward pass verifies them, and accepted tokens are emitted in bulk, substantially increasing throughput with no accuracy loss.

### Why a sidecar

Standard quantization pipelines (including mlx-lm's `sanitize()`) strip `mtp.*` tensor keys at conversion time. The MLX-quantized checkpoint you download from Hugging Face typically does not contain MTP weights.

SWP solves this by re-extracting the `mtp.*` tensors from the original BF16 checkpoint, quantizing only those tensors, and publishing the result as `mtp.safetensors` to a dedicated sidecar repo on Hugging Face. This is the same pattern Skulk uses for vision encoder weights.

### How it works

When a model card declares `mtp_heads = true` and `mtp_sidecar_repo`, Skulk:

1. Downloads `mtp.safetensors` from the sidecar repo alongside the base model weights.
2. Loads the sidecar at model load time and makes the weights available to the runner.
3. Uses the MTP heads during generation for speculative decoding (on runners that support it).

If the sidecar is declared but the file is not found locally, Skulk logs a warning and continues with standard autoregressive generation.

### Models with native MTP heads

The shipped cards that declare an MTP sidecar are the Qwen3.5 and Qwen3.6
quantizations:

- `mlx-community/Qwen3.5-2B-4bit`
- `mlx-community/Qwen3.5-9B-MLX-4bit`
- `mlx-community/Qwen3.5-27B-4bit`
- `mlx-community/Qwen3.6-27B-4bit`

Gemma 4 does **not** use native MTP heads; it uses an external drafter model (`assistant_model_repo`) for speculative decoding, which is a separate feature.

### Adding MTP to a card

```toml
[runtime]
mtp_heads = true
mtp_max_depth = 1
mtp_sidecar_repo = "FoxlightAI/qwen3-5-9b-base-mtp"
mtp_sidecar_revision = "06c840b3529f5695648807d993b1cb48b576a988"
```

The sidecar repo must be published by SWP before adding these fields. See the [SWP documentation](https://foxlight-foundation.github.io/skulk-weights-publisher/) for how sidecars are produced and published.

### Speculation on the served (GGUF) engine

The sidecar path above is MLX-only. GGUF cards that run under the served
`llama-server` engine declare speculation through a different set of `[runtime]`
fields: `served_spec_type`, `served_spec_n_max`, `served_spec_draft_repo`, and
`served_spec_draft_file`. See [Speculative Decoding](speculative-decoding) for
how those fields map to the served backend.

## Declarative vs Resolved

The model card is the **declarative** capability source.

At runtime, Skulk resolves the card plus tokenizer/model-family facts into a normalized execution profile. That resolved profile is what prompt rendering, reasoning defaults, and output parsing consume.

That gives Skulk three good properties:

- old cards remain valid
- advanced cards unlock refined behavior
- runtime code can rely on normalized values instead of ad hoc optional checks

## Resolution Precedence

When Skulk resolves a runtime capability profile, it uses this order:

1. explicit advanced model-card declarations
2. conservative family/model heuristics
3. generic fallback behavior

That means a custom or built-in card can refine behavior without breaking old
cards that only declare coarse metadata.

## Extended Card Example

This is a minimal example of a custom card that opts into refined runtime
behavior:

```toml
model_id = "custom/gemma-compatible"
n_layers = 10
hidden_size = 1024
supports_tensor = false
tasks = ["TextGeneration"]
family = "gemma"
capabilities = ["text", "vision", "thinking"]

[storage_size]
in_bytes = 1073741824

[reasoning]
supports_toggle = true
format = "channel_delimited"
default_effort = "medium"
disabled_effort = "none"

[modalities]
supports_native_multimodal = true

[tooling]
tool_call_format = "gemma4"

[runtime]
prompt_renderer = "gemma4"
output_parser = "gemma4"
```

The card stays declarative. Skulk still resolves it into a normalized runtime
profile before execution code consumes it.

Speech cards use the same pattern:

```toml
model_id = "custom/kokoro-tts"
n_layers = 1
hidden_size = 1
supports_tensor = false
tasks = ["TextToSpeech"]
family = "kokoro"
capabilities = ["tts"]

[storage_size]
in_bytes = 1073741824

[placement]
compatible_backends = ["mlx_audio", "mlx_audio-metal"]
backend_preference = ["mlx_audio-metal", "mlx_audio"]

[audio]
kind = "tts"
default_response_format = "mp3"
response_formats = ["mp3", "wav"]
# Set true only after the Skulk runtime validates this model/backend streaming path.
supports_streaming = true
supports_realtime = false
supports_voice_listing = true
voices = ["serena", "ryan"]
default_voice = "ryan"
supports_reference_audio = false
sample_rates = [24000]

[[audio.voice_catalog]]
id = "serena"
name = "Serena"
preferred_languages = ["zh"]

[[audio.voice_catalog]]
id = "ryan"
name = "Ryan"
preferred_languages = ["en"]
```

For a reference-conditioned voice, the catalog ID and `reference_profile` must
match and the card must declare `supports_reference_audio = true`:

```toml
[[audio.voice_catalog]]
id = "angus"
name = "Angus"
preferred_languages = ["en"]
reference_profile = "angus"
```

The central profile manifest pins the local MP3 digest and exact transcript.
Model cards intentionally repeat the public voice order so API and dashboard
behavior remain explicit model truth; CI verifies every bundled cloning card
against the central manifest.

## When to Extend a Card

Extend a card when:

- the model needs special prompt rendering
- the model uses a non-generic reasoning format
- the model supports modalities or controls that generic metadata cannot express
- the dashboard or API needs to expose richer behavior safely

For concrete examples, see [Model Capabilities](model-capabilities) and the per-family notes in [Model Behaviors](model-behaviors/gemma4).
