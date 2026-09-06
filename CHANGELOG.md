<!-- Copyright 2025 Foxlight Foundation -->

# Changelog

This project records release notes here and mirrors public-facing notes in
`website/docs/release-notes/`.

## [Unreleased]

### Added

- Muse Glimmer (Meta, August 2026) is a first-class model family on every
  serving lane. The capability resolver now derives the family's wire
  contract from the card family or model id, the same way it does for
  Gemma 4 and gpt-oss, so a registry or auto-imported card with no explicit
  tooling or runtime sections resolves to always-on channel-delimited
  reasoning (no toggle, default strength high), ATEM tool calling, and the
  new `muse_glimmer` output parser. The MLX engine gains a streaming parser
  for the `to=self` / `to=user` / `to=<tool>` channels and Meta's ATEM
  markup, the shared text dialect reader and the no-tools scaffolding scrub
  learn ATEM, and every engine's request path translates `reasoning_effort`
  onto the template's `reasoning_strength` kwarg (`none`/`minimal` become
  `low`; `xhigh` is honored). Model cards gain `runtime.vllm_reasoning_parser`
  (explicit only, like the tool-call parser), mapped to vLLM's
  `--reasoning-parser` so a card can pin `muse_glimmer` for both. The
  in-process llama.cpp engine is not enabled for this family: the pinned
  llama-cpp-python binding vendors a llama.cpp build that predates the
  architecture.

### Fixed

- `enable_thinking` is honored on the in-process llama.cpp engine for text
  models whose GGUF chat template reads it (the Qwen3-family shape). The
  binding's `create_chat_completion` offers no template-kwarg channel, so
  the control was silently ignored and a thinking-default model reasoned on
  every request regardless; the runner now wraps the default Jinja
  formatter with a per-request slot. Guessed family formats, vision
  handlers, and templates without the control are untouched, and an
  unexpected library shape degrades loudly to the previous behavior. The
  engine's reasoning parser follows the toggle: a thinking-off prompt
  pre-closes the think block, so the generation starts outside it, and the
  parser no longer assumes the mid-reasoning start (which misrouted the
  whole plain answer into `reasoning_content` and starved tool recovery of
  its visible text).

- A `tool_choice` that forces a function name matching none of the offered
  tools (or arrives with no tools at all) is now rejected with a 400 at the
  API boundary, on every engine. The in-process engines never see
  `tool_choice`, so such a request previously answered from the full tool
  list with no report of the mismatch, and only the served engines surfaced
  the caller's error.

- A plain JSON answer from an unmarked-dialect model (Llama on MLX) streams
  incrementally again. The message-opening brace provisionally opens a
  tool-call block whose closing token is a generation stop that never arrives
  as text, so the whole answer was held until the terminal chunk and
  time-to-first-token grew to the full generation time. The open is now
  provisional: the buffered prefix is released the moment it can no longer be
  a call, bounded by the first decisive key, and a real call still parses
  exactly as before.

- A quoted argument containing the dialect's own closing marker (an
  HTML-writing tool passing `"</tool_call>"`) no longer truncates the block
  and errors the generation, on the streaming paths whose block interior is
  known: Gemma 4's quoting and templates that render arguments as JSON.
  Interiors with other quoting rules (Qwen3 XML) keep the previous scan.

- Chained Llama `<|python_tag|>` calls are no longer split inside a quoted
  argument: a semicolon in a shell command, SQL, or prose is data, and the
  chained objects are now read as successive balanced JSON spans.

- Visible text a model writes around its call is delivered as content
  alongside the tool calls, instead of being swallowed with the markup, for
  the dialects that know where their markup ends: the unmarked call object
  and the Mistral `[TOOL_CALLS]` array, on both the MLX streaming path and
  the llama.cpp text-recovery path. Mistral's displaced upstream
  `NAME[ARGS]` form keeps parsing through the inner parser, with no
  remainder.

- Gemma 4 models can now call tools on the in-process llama.cpp engine. The
  engine's bundled chat handler does not parse Gemma 4's call format, and the
  text-recovery path had no dialect for it, so well-formed calls streamed to
  the caller as raw markup (observed live on the GGUF card with tools
  offered). The shared text parser gains the dialect, backed by the same
  implementation the MLX engine already used, so both engines read it
  identically.

- An enabled model store with a blank `store_host` or `store_path` is now
  refused loudly at config validation instead of running crippled. The blank
  shape matches no node, so no store server ever started while every client
  built unusable `http://:12415` URLs, and the resulting failure was
  misclassified as "model not in store", starving downloads against a host
  that can never answer instead of taking the direct Hugging Face fallback.
  The refusal covers node startup and the Settings save (the dashboard also
  blocks the save with a visible message), and a URL that cannot even be
  requested now classifies as store-unreachable immediately, with no retry
  delay, so the fallback engages even if a bad address reaches the client
  through another path.

- The Mistral tool-call dialect now works on the MLX engine. The shared text
  parser has read `[TOOL_CALLS]` arrays since they were added, but no MLX
  tokenizer wiring selected it: Mistral templates carry no `<tool_call>`
  marker, so requests fell through every wiring branch and the model's calls
  leaked as content. Templates speaking `[TOOL_CALLS]` now wire the Mistral
  whole-block parser, closed at end of generation the way the unmarked
  dialect closes. A bundled card for a small Mistral
  (`mlx-community/Ministral-8B-Instruct-2410-4bit`) makes the dialect
  live-testable on a 24GB node.

- The Qwen3.6 FP8 cards now enable tool calling on the vLLM engine. Qwen3.6
  emits the XML function format, which vLLM's `qwen3_xml` parser reads; the
  parser name was validated live on an A100-class GPU with the full served
  tool suite on both cards, which is what the cards' own deferral note asked
  for before pinning.

- The no-tools marker protection now covers every engine. The scan below was
  in-process MLX only: the served engines' servers never parse without tools
  in the request, and the llama.cpp recovery branch is likewise skipped, so a
  model that wrote a call anyway leaked its dialect markers to the caller as
  content (observed live with a gemma card and `tool_choice: "none"`). The
  `llama_server`, `vllm`, and `llama_cpp` runners now stream no-tools content
  through a shared scaffolding scrub that removes the cross-dialect marker
  vocabulary, holding partial markers across chunk boundaries.

- A request offering no tools still has its markers stripped. Skipping the scan
  entirely when none were offered, which is what keeps `tool_choice: "none"`
  from producing a call, also meant nothing recognized a block the model wrote
  anyway, so its markers went straight to the caller. The block is now always
  recognized; whether it may become a call is what depends on the request.

- A tool call handed back as content no longer carries the model's control
  tokens. When a call names no offered tool it is delivered as text so the
  caller can see what the model did, but the block was handed back verbatim, so
  `<|python_tag|>` and `<tool_call>` markers ended up in the answer. The
  markers are stripped from an answer; a response already flagged as an error
  still carries the raw block, since there it is the evidence of what was
  malformed.

### Fixed

- gpt-oss and DeepSeek V3.2 no longer return a tool the caller never offered.
  Those two families parse their calls out of the token stream themselves and
  are selected before the marker path, so the offered-tools rule never saw
  them: a gpt-oss request sending `tool_choice: "none"`, which removes the
  tools, still came back with a call, and its name carried the model's own
  namespace prefix. Their output now passes through the same rule, and a
  rejected call is delivered as content so the caller sees what the model did
  rather than a blank answer.

### Fixed

- A model's parallel tool calls all reach the caller. Several families write
  each call in its own block, and the stream consumer stops at the first chunk
  carrying a finish reason, so one response per block delivered the first call
  and dropped the rest. The calls of every block in a message are now coalesced
  into a single response carrying a `tool_calls` array, which is the shape
  OpenAI clients expect, and any text after the calls is released without a
  finish reason so the tool response stays the terminal chunk.

### Fixed

- Reasoning no longer hides a tool call on the MLX engine. Tool parsing runs
  downstream of the thinking parser, so a model that reasons before calling a
  tool sent its reasoning through the tool parser first; that text decided the
  message was not a call, and the marker that followed was never examined, so
  the caller received the raw markup as content. Reasoning chunks now pass
  straight through without taking part in that decision. This also means a call
  a model only contemplated inside its reasoning is no longer executed, matching
  the behavior the llama.cpp engine already had.

### Fixed

- `tool_choice` is now honored on the in-process engines. Only the served
  engines forwarded it to a server that acts on it, so an MLX or llama.cpp
  model ignored it entirely: a request sending `"none"` and asking for the tool
  by name returned the tool call on every attempt. The option is now applied
  before dispatch, so it means the same thing on every engine. `"none"` removes
  the tools from the request, and naming a single function narrows the offered
  tools to that one so the model cannot call a different tool than the caller
  asked for. `"required"` remains a best-effort instruction on the in-process
  engines, since forcing a call there would need constrained decoding.

### Fixed

- Tool calls whose markers arrive split across chunks are now recognized. A
  generation chunk is whatever the streaming detokenizer could resolve that
  step, not a token, so an opening marker that is a single token id still
  reaches the parser in pieces: `<tool`, `_`, `c`, `all>`. The parser tested
  each chunk on its own, so for most models the block never opened and the
  caller received the raw markup as message content with a `stop` finish
  reason. Observed on a Qwen model served by the MLX engine, where the model
  emitted a perfectly well formed call. Text is now scanned across chunk
  boundaries by carrying forward only the trailing run that could still become
  a marker, and the closing marker is matched against the accumulated block.
  That run is shorter than the longest marker, so ordinary answers stream with
  at most a few characters of latency, and the scan keeps looking after
  ordinary text has been released, so a model that writes a sentence before
  calling ("I'll check that.") still has its call recognized. The unmarked
  dialect opens on a brace, which also appears in prose, so there a call is
  recognized only at the start of the message; its distinctive marker still
  opens one anywhere. Text the model writes after closing a call is delivered
  rather than swallowed into the block, and a second call in the same message
  is recognized.

### Fixed

- Tool calling now works for Llama models on the MLX engine, and the shared
  text parser recognizes the dialects the other families write. Llama declares
  only its end-of-turn token as a stop token, not `<|eom_id|>`, which is how it
  ends a message that hands off to a tool, so generation ran past the end of
  the call and wrote the next turn's header into the answer text. Llama also
  writes the call as a bare JSON object with no opening marker, so nothing
  recognized it as a call at all: a caller offering a tool received JSON in
  `content`, `finish_reason` of `stop`, and no `tool_calls`. Skulk now stops at
  the message boundary for any model whose vocabulary has that token, and reads
  the whole block with a set of cross-family dialects covering Llama
  `<|python_tag|>` calls, Mistral `[TOOL_CALLS]` arrays, GLM
  `<arg_key>`/`<arg_value>` pairs, and an unmarked call object that is the
  entire message, alongside the harmony channels and `<tool_call>` blocks
  already supported.

- A model reaching for one of its own built-ins no longer surfaces as a tool
  call. Llama answers some plain questions with a call to `print`, and gpt-oss
  has `python` and `browser`; a caller has no implementation for those names,
  so a response naming no offered tool is now returned as ordinary content. A
  request that declares no tools may never receive `tool_calls`: the response
  is still scanned so a recognized block's dialect markers are stripped rather
  than delivered, but nothing comes back as a call, so a model writing
  something call-shaped, which is what a request asking for JSON output
  invites, cannot return `tool_calls` to a caller who offered none. Relatedly, text that opens
  like a call but does not parse as one, which is what a model answering in
  JSON looks like when tools are also offered, is returned as content instead
  of being reported as a generation error.

### Fixed

- Chat completions never return an empty body, and streaming responses always
  terminate. A task that ended without producing any output, for example after
  being cancelled, previously tripped an assertion inside the response
  generator; because the status is committed before the body streams, callers
  received HTTP 200 with zero bytes and every OpenAI-compatible client failed
  while parsing rather than reporting the real problem. The non-streaming path
  now returns the standard error object as its body, and the streaming path
  emits an error frame followed by `data: [DONE]` instead of closing without
  a terminator, which a client cannot distinguish from a dropped connection. A
  turn that produced text but never reported a finish reason is also treated
  as a failure rather than returned as a silently truncated completion.

### Fixed

- Streaming chat completions now carry `"object": "chat.completion.chunk"`.
  Every SSE frame previously carried `"chat.completion"`, the non-streaming
  discriminator, because one response model served both paths. Clients that
  read `choices[0].delta` directly were unaffected, which is why this went
  unnoticed, but clients that validate the discriminator reject such a stream
  outright, including the Vercel AI SDK's openai-compatible provider. The
  streaming and non-streaming responses are now separate models so the two
  cannot drift again. This changes bytes on the wire for streaming responses,
  toward the documented OpenAI format rather than away from it.

### Fixed

- Model execution authorization now follows the action that introduced the
  exact card. Signed registry publication authorizes repository code for every
  provenance class, explicit external-model addition authorizes its pinned
  card, and bundled cards remain authorized by the Skulk release. Hugging Face
  additions that omit a revision resolve `main` once to a full immutable
  commit. Read and launch paths no longer fetch or persist unknown Hub cards as
  a side effect, and caller-specified exact placements must match current
  catalog truth rather than merely reuse its alias. The dashboard no longer
  exposes the redundant Model trust ceremony;
  its historical config, state, wire fields, and endpoints remain deprecated
  and inert for rolling compatibility. Historical executable custom cards with
  no immutable revision fail closed until re-added, and ordinary model-add
  responses now wait for their exact ordered catalog mutation before returning.
  Image, embedding, and speech inference endpoints translate an unknown catalog
  alias to HTTP 404 instead of leaking the strict lookup failure as HTTP 500.
  The elected master also revalidates quick and exact placement cards against
  command-ordered catalog truth, closing replacement/deletion races after an
  API node has prepared a placement.
  Executable bundled fallback cards must pin an immutable source revision, and
  installed custom-card sidecars no longer recreate catalog authorization after
  the operator deletes the custom card.
  Separate processor, vision-weight, assistant, and speculative-draft
  repositories must also carry their matching immutable revisions.
  The low-level explicit-download endpoint now requires operator authority and
  rejects shard cards that do not exactly match authorized catalog truth.
  Authorization comparison ignores only the signed snapshot publication stamp.
  Signed-card, revision, installed-sidecar, and artifact-manifest verification
  still fail closed.

- Pre-publication qualification cleanup now names the complete temporary card
  it owns, and the elected master compares that exact card before deleting the
  alias. An older or retried job can no longer remove a newer qualification
  replacement that reused the same model ID.

- Same-artifact signed card replacements now run the card-only installed-sidecar
  refresh before a staged-cache fast path can report the model ready. This
  prevents a newly approved replacement card from passing placement and then
  failing runner startup against the prior installed identity, without
  retransferring unchanged model bytes.

### Added

- Authenticated operator workflows can now install one complete pinned model
  card through `POST /models/add-card` for pre-publication qualification. Skulk
  preserves the exact artifact bundle while stripping registry identity and
  provenance, so testing cannot impersonate signed registry truth; the explicit
  exact-card add authorizes its pinned repository code. Headless registry automation
  may use the narrowly scoped `SKULK_EXACT_CARD_QUALIFICATION_TOKEN` for only
  this immutable temporary install and server-owned custom-card cleanup
  lifecycle; only service-authenticated installs receive the ownership marker,
  and the credential cannot replace or delete any other card. The elected
  master rechecks that precondition at the serialized ordering boundary, and
  success waits for local persistence of the indexed event carrying the exact
  originating command ID. Cleanup preserves downloaded artifact bytes without
  allowing their temporary installed sidecar to re-enter the catalog, and later
  signed-registry refreshes update the master's ownership guard. Qualification
  downloads additionally pin the immutable v2 bundle identity through both the
  API node and canonical store, preventing a later alias replacement from
  redirecting the bytes under test.

- The dashboard has a new Integrations page that generates ready-to-paste
  configuration for connecting external tools to the cluster: Claude Code,
  OpenCode, Codex, Hermes, OpenClaw, Pi, AnythingLLM, Open WebUI, n8n and
  Firefox. Snippets are built from live cluster state rather than being static
  examples, so they carry the ids of models that currently have a ready
  instance, those models' real context windows, and their capability flags
  (image input is declared for vision models, and models that mark their
  reasoning are configured to send it back on later turns). The address in a
  snippet is the node's routable address rather than `localhost`, with a
  chooser between the local network and Tailscale when both are available, and
  Docker recipes rewrite it to `host.docker.internal`.

- Signed registry-v2 cards can now describe one exact immutable artifact bundle
  inside a shared upstream repository. Skulk downloads only the required file
  allow-list, verifies sizes and available upstream object identities, preserves
  directory layout, loads engines from the declared artifact root, and retains
  bundle identity in installed sidecars and store generations. Existing v1
  cards retain their previous behavior.

- Served GGUF vision now uses one truthful model card for the base quant,
  immutable projector, vision capability, and native MTP behavior. New cards
  pin one exact projector path and size; downloads retain only that projector,
  the runner verifies it against the installed manifest before launching
  `llama-server --mmproj`, and CPU placement disables projector offload. Vision
  plus MTP degrades to serial serving until concurrent multimodal qualification.
  Homogeneous CUDA, ROCm, and Vulkan RPC placements reserve the projector on
  their selected driver and route image media only there; legacy cards continue
  through the in-process llama.cpp path.

- Placement now keeps heterogeneous engine choice planner-owned: model cards may
  declare an open set of compatible backends and an ordered fallback preference,
  and the planner automatically falls through when an earlier engine or host is
  unavailable. Repository-code trust is now one operator decision per immutable
  model-card identity in cluster Settings, synchronized to the canonical store
  and every node rather than repeated machine by machine. Previews expose stable
  failure categories and placement responses add `X-Skulk-Placement-Failure`
  without replacing their readable error message. Trust and custom-card writes
  require direct loopback or authenticated operator-gateway access; synchronized
  settings preserve node-local Hugging Face credentials and atomically retain
  owner-only config permissions. Trust changes are serialized by the elected
  master and replicated as durable state rather than competing replaceable
  config snapshots, so concurrent approvals and revocations cannot overwrite
  one another or be resurrected by an unrelated Settings save.

- The managed llama.cpp served engine advances to b10434 across the CUDA and
  Vulkan wheels, verified Linux archives, and the prebaked CUDA pod image. The
  release adds the RPC tensor operation required for DeepSeek V4 multi-node
  execution, Qwen 3.8 text and native long-context support, recurrent-state
  rollback, and served reasoning-effort plumbing. Every bundled
  `ggml-rpc-server` advances with `llama-server` because the intervening RPC
  protocol changed; AMD continues through the fleet-qualified Vulkan lane
  because upstream b10434 publishes no Linux ROCm archive. DeepSeek V4 may now
  use served CUDA, while its independently versioned in-process CUDA backend
  remains excluded.

- The managed CUDA llama-server wheel now ships for Linux aarch64 as well as
  x86_64. The aarch64 lane is built natively with CUDA 12.9 for compute
  capability 12.1, allowing Grace Blackwell and GB10 nodes to use the CUDA
  served engine instead of falling back to Vulkan.

- Supervised startup now preserves an installed managed CUDA or Vulkan
  llama-server wheel across its routine `uv sync`; previously the exact sync
  could prune the installer-managed wheel on the first service restart and
  silently return the node to tarball provisioning.

- Intelligent Fabric now speaks and identifies as Skulk rather than presenting
  a separate Steward character. The dashboard streams answer prose through a
  ready TTS model only when its voice catalog contains the signature `skulk`
  voice, pins that voice on every sentence, and keeps speech failure isolated
  from authoritative text generation. Internal steward route, role, and model
  identifiers remain compatible for existing clients.

- Skulk now ships its signature voice: a new bundled reference profile named
  Skulk, constructed synthetically like the existing ten and paired with the
  same shared conditioning transcript. It appears in the voice catalog of
  every validated cloning card and replaces Kite as the default voice, so any
  synthesis request that does not choose a voice — including speech spoken on
  behalf of the steward — speaks as Skulk. The reference-voices README now
  also records how bundled profiles are authored.

- Model artifacts are now self-describing and air-gap durable: canonical and
  staged copies retain their complete model card, immutable selection,
  verification state, companion ownership, and SHA-256 manifest in an atomic
  installed-card sidecar. The authoritative store automatically inventories
  existing node caches and can import missing artifacts through resumable,
  capability-bound peer transfers without returning to Hugging Face. The Model
  Store dashboard exposes cache placement, verified versus local-legacy truth,
  reconciliation progress, available updates, and signed warn-only advisories.
  Durable deletion tombstones prevent missed node-cache evictions from
  resurrecting removed models or their companion artifacts. Partial legacy
  directories are never promoted from a name match, resumable peer imports
  account only for missing bytes, capability exports enforce their transfer
  ceiling, artifact eviction immediately clears stale installed state, startup
  polling survives the delayed first scan, and companion recovery selects the
  generation belonging to the current signed owner card. Internal peer imports
  reject proxy-forwarded loopback requests, while omitted immutable card IDs
  retain their documented current-generation compatibility behavior.

- The dashboard voice loop now narrates code blocks: when a fenced code block
  streams during live generation, the assistant voice speaks a short opener,
  occasional fillers while the block streams, and a closer when it ends,
  instead of leaving dead air where the unspoken code would be. Fillers fire
  only when the voice has actually run out of queued speech, so narration can
  never stack behind prose or chatter on a fast stream; adjacent blocks
  continue without a false finish, replayed messages stay silent about code
  by construction, and the Narrate code toggle beside Auto speech turns the
  behavior off.

- Intelligent Fabric mode gives the cluster a resident steward: an assistant
  the fabric itself keeps placed as a hidden system instance, ready to answer
  operator questions about cluster health, models, downloads, and diagnostics.
  The steward investigates through read-only tools before answering and
  cannot change the cluster. It is off by default; enabling it in Settings
  makes the master establish the placement as a planner invariant — placed on
  the best available nodes from a benched preference list (with a GGUF
  universal floor so every fleet can serve one), re-placed after node loss or
  master failover, and protected from ordinary deletion. Clients talk to it
  through the standard OpenAI-compatible chat surface as the virtual model
  `skulk/steward` (streaming included, no steward-specific client code), poll
  readiness at `GET /v1/steward`, and see it flagged with
  `system_role: "steward"` in `GET /v1/models` so pickers can badge or
  separate it. The dashboard gains a Skulk fabric-chat page, chat-middleware extensions
  run on steward turns exactly as on ordinary completions, and a steward that
  is still being placed answers with a clean 503 status payload instead of
  failing mid-answer.
  The resident now converges upward when better capacity appears: an improved
  brain must remain placeable for five minutes, stages before replacement, and
  waits for a 30-second idle window. The parser-pinned 35B FP8 vLLM card joins
  the default tier; explicit `NodeResources.api_available` telemetry elects one
  canary owner while still covering `--no-api` worker hosts;
  and node-scoped diagnostics include each node's doctor findings.

## [1.5.1] - 2026-08-30

### Changed

- Published the current stable `main` runtime through one coordinated desktop
  release: a signed and notarized Apple Silicon app, direct `amd64` and `arm64`
  Debian packages, the signed Foxlight APT repository, and the Homebrew cask
  all carry the same Skulk version and exact source provenance.
- The macOS menu-bar app now checks the canonical stable release manifest and
  offers the signed DMG when a newer stable version is available.
- The APT release gate now proves a clean `sudo apt install skulk` journey on
  both supported Linux architectures without starting or joining a cluster.

Skulk runtime and wire behavior are unchanged from 1.5.0; this patch release
provides a stable version boundary for the completed desktop distribution path.

## [1.5.0] - 2026-08-07

### Changed

- Redesigned the dashboard's dark mode around the Foxlight operator design
  system's Den palette, replacing the previous high-contrast scheme: indigo
  surfaces over a deep night canvas, a starlight accent for everyday
  interaction, and amber reserved for work actually in flight (RAM a model is
  holding, downloads in progress, attention badges). Light mode is unchanged.
  Both palettes share one token vocabulary, so components never branch on the
  theme name. Building the dashboard with `VITE_NIGHT_SKY=1` additionally
  crowns dark mode with the brand valley's star field (occasional shooting
  stars included, and the abstract mesh stands down); the default build ships
  a CSS-only night gradient with the mesh.

### Added

- Bundled ten checksummed English reference voices (Angus, Ember, Hannah, Ian,
  Jake, Kite, Rufus, Samson, Sydney, and Sylvie) for the validated Qwen Base,
  LongCat, and Fish voice-cloning cards. They appear through the ordinary voice
  catalog and resolve to local conditioning audio plus its exact transcript only
  inside the selected worker, with Kite as the shipped default. Qwen Base now
  ships the stable six-bit conversion; the unstable 0.6B CustomVoice and
  four-bit Base cards are no longer offered.

### Fixed

- Dashboard speech now prepares model turns for synthesis structurally rather
  than sentence-by-sentence alone: an unpunctuated block such as a bold story
  title gains terminal punctuation so it no longer bleeds into the following
  sentence, Markdown block and heading boundaries end a spoken segment even
  when the model omitted punctuation, emoji are stripped instead of being
  read aloud, and a horizontal rule becomes a brief audible pause on the
  streaming playback timeline instead of spoken dashes.

- Dashboard speech now sends completed turns and replay requests to batch-only
  TTS models as one synthesis call. Sentence-sized request queues remain
  reserved for cards that truthfully advertise streaming PCM, avoiding the
  repeated full-generation overhead that made LongCat and Fish playback
  unnecessarily slow.

- Supervised Linux updates now rebuild the dashboard with Skulk's bundled
  Node.js runtime. The launchd/systemd startup wrapper previously used bare
  `npm` even though the official installer uses the required pinned runtime.
  Linux nodes without a system Node.js installation therefore kept serving an
  old dashboard after successful code updates while logging non-fatal npm
  failures. Boot-time prep now reserves headless mode for explicitly API-only
  nodes and retains system npm only as a recovery fallback.

- Dashboard speech now sends the same deterministic seed for every generated
  sentence and replay segment. The public speech API and built-in TTS provider
  also accept an optional unsigned 32-bit seed, which the speech runner applies
  immediately before model generation; callers that omit it retain upstream
  advancing-random-stream behavior.

- Dashboard realtime STT Auto-send now submits final transcripts through the
  same chat-completions path as typed prompts. Voice turns retain the complete
  dashboard conversation, use normal generation limits, and share the same
  streaming, cancellation, and sentence-paced TTS behavior instead of creating
  a separate socket-local conversation capped at 256 output tokens.

- Omitted TTS `max_tokens` no longer inherits undersized upstream model
  defaults that hard-cut ordinary speech mid-word. Speech runners now apply a
  4096-token serving budget only to models that explicitly declare the control,
  while preserving caller-supplied limits and leaving unsupported models alone.

- The dashboard model-store page now retries transient registry and download
  request failures until it has both a successful registry snapshot and no
  active downloads. A brief API connection reset during a fresh-install model
  download can no longer leave the page stuck at "0 models in store" after the
  model was registered successfully.

- Disabled the Qwen3.5 2B MLX card's MTP sidecar after clean-install text and
  vision journeys showed repetitive generation until the output limit. The
  same shipped model now uses vanilla decoding, which completes both journeys
  normally while retaining its text, thinking-toggle, and vision capabilities.

- Fresh multi-node installs now converge their per-node bootstrap model stores
  on the elected master's routable store endpoint. Followers retry
  authoritative config sync through the startup window, stop superseded local
  store servers, and update dashboard and worker clients together, preventing
  dashboard download followed by placement on another node from downloading
  the same multi-gigabyte model from Hugging Face twice.

- Routed text-only requests on native MLX-VLM models through their language
  model instead of the multimodal outer model. This prevents Qwen3-VL text
  chats from producing long corrupted repetition while preserving the native
  image-generation path.

- Prevented a fresh node's startup download-progress scan from exhausting
  Hugging Face API limits by fetching metadata for every shipped model card.
  The scan now validates only models that have local files to resume or
  recover, so a user's first selected model receives the available request
  budget.

- Realtime STT admission no longer reports a ready runner as overloaded while
  cache-miss download lifecycle state is still converging across API nodes.

- Prevented the dashboard Settings panel from crashing when it reads the
  sparse `model_store` configuration generated by a fresh installation.

- Stopped unreachable per-interface IPv6 link-local probe candidates from
  flooding normal logs with misleading peer-down warnings.

- Fixed fresh Apple installs returning HTTP 500 for MP3, FLAC, OGG, and Opus
  speech output because `mlx-audio` expected an external `ffmpeg` executable.
  Skulk now ships a platform encoder dependency and exposes its bundled binary
  to speech runners when no system `ffmpeg` is installed.

- Fixed store-backed model launches exhausting a worker's filesystem while
  recently used staging data was still inside the 40 GiB warm-cache budget.
  Store transfers now serialize exact capacity admission with the byte
  transfer, include base and companion repositories, count resumable manifest
  bytes, and treat same-filesystem hardlinks as zero allocation. Skulk protects
  every active model transaction and live runner, evicts only idle copies in
  least-recently-used order until the exact additional allocation fits, and
  preserves 10 GiB of operating-system headroom. Canonical model-store
  downloads use the same serialized exact-byte admission without ever evicting
  authoritative artifacts, and direct Hugging Face fallback applies it to the
  actual model-cache filesystem instead of the unrelated staging cache. If any
  destination cannot meet its target, the placement receives an actionable
  failure before any more bytes are written.

- Kept fixed-window llama.cpp contexts at the safe 8192-token floor on
  unified-memory AMD APUs. Placement still uses their combined VRAM/GTT pool,
  but no longer misclassifies that pool as discrete VRAM and lets a large
  startup KV allocation OOM the node and its co-hosted model store. The shipped
  systemd unit also contains any future runner OOM to the child process instead
  of stopping the entire Skulk service.

- Fixed the documented launchd/systemd install step failing immediately after a
  successful one-command install because `uv` lived under `~/.local/bin` but
  the parent shell had not reloaded its PATH. Both service installers now
  resolve the same user-space tool locations as the runtime wrapper.

- Fixed fresh-install model-store startup failures caused by the old `58080`
  listener racing normal outbound connections in operating-system dynamic
  client-port ranges. The runtime, installer-generated config, dashboard, and
  docs now use the explicit `12415` default; existing configurations that set
  `store_port` remain unchanged.

- Fixed Fish Audio S2 synthesis returning speech unrelated to the input by
  pinning a minimal `mlx-audio 0.4.3.post1` maintenance carry of the upstream
  hidden-state generation fix.

- Fixed HTTP model-store staging progress to use the canonical registry byte total across fresh and resumed multi-file transfers, restoring the bounded fraction gate that prevents progress telemetry from flooding the ordered control plane (#520).

### Changed

- Redesigned the dashboard's Find Models dialog. Catalog rows now lead with the
  card's human-readable model name over a family monogram tile, show capability
  chips shared with the store table plus size and context metadata, and label
  store state with an explicit "In store" chip instead of bare check and arrow
  glyphs. Downloads start only from a labeled Download button (per quantization
  in the expanded size list), so clicking a row can no longer silently kick off
  a multi-hundred-gigabyte transfer. The family sidebar dropdown became a chip
  rail, and Hugging Face results gained author and popularity metadata lines.
  Rows carry the model's brand mark (or a monogram when no vector or bundled
  raster mark exists), quantization and artifact-format tiles, and a
  fleet-first ordering: models the local fleet can serve list first, and
  models needing more capacity move to a "Needs burst capacity" section with
  an amber Burst chip explaining whether size or artifact format exceeds the
  fleet. Burst rows stay fully downloadable; Hugging Face size verdicts prefer
  exact GGUF artifact sizes and metadata parameter counts, falling back to
  name-derived estimates marked as such. The Hugging Face trending list now
  sorts by Hugging Face's trending score instead of all-time downloads, gains
  a "Show more" pager, and rows surface task chips, gated-license markers,
  parameter counts, artifact sizes, and context lengths from the enriched
  `/models/search` response. Results also carry derivation lineage
  (finetune, quantized, merge, adapter, shown as a small classification
  tile on the row), tagged papers, languages, and architecture, and every
  discovery row links its exact Hugging Face repository. The row's info
  popover became a real dossier: it lazily fetches the model card's own
  description through the new `GET /models/card-summary` endpoint and shows
  lineage with a link to the parent repository, architecture, languages,
  license, and arXiv papers. A task chip rail on the search tab filters
  trending and search results by Hugging Face task (text, vision, STT,
  TTS, embedding, image generation) via the endpoint's `pipeline_tag`
  parameter. GGUF results expand into a per-quantization download chooser
  backed by the new `GET /models/gguf-quants` endpoint, and default GGUF
  selection now ranks companion artifacts (speculative drafters such as
  dspark/dflash files, imatrix calibration data) behind every real quant,
  so adding a repository can no longer silently stage a 10 GB drafter
  wearing the model's name.

- **Concurrent slots on the served llama.cpp engine no longer shrink each
  request's context window.** `SKULK_LLAMA_SERVER_PARALLEL` asks a node to serve
  N generations at once. Until now that came with a hidden cost: llama.cpp gave
  each slot only an equal share (`n_ctx / N`) of the model's context window,
  while Skulk's API kept advertising and admitting against the full window, so a
  prompt that fit could be truncated by the engine rather than refused. Skulk
  compensated by silently capping the requested slot count, which meant an
  operator asking for 8 slots could quietly get 2. The runner now launches the
  server with a unified KV cache above one slot, which gives every slot the
  whole window at no extra memory cost, and the declared slot count is honored
  exactly with no cap. Fresh installs now use the release-qualified 16-slot
  width instead of silently serving every request serially; an explicit
  `SKULK_LLAMA_SERVER_PARALLEL=1` retains the prior serial behavior. Above one
  slot, the slots share a single pool instead of holding private shares. Before
  each generation Skulk asks llama-server for the exact rendered prompt length,
  reserves that input plus the bounded maximum output, and queues until the sum
  fits. Reservation waiters are admitted FIFO so sustained short traffic cannot
  starve an earlier long request. A failed token-count probe reserves the whole
  pool and runs alone. This preserves real concurrency for bounded requests
  without allowing aggregate long-context traffic to terminate the server
  (#689).

- **The service template no longer pins a cluster namespace.** The installed
  `skulk.env` template used to set `SKULK_LIBP2P_NAMESPACE=foxlight-main`,
  which put every template-based install on one shared namespace while nodes
  launched manually with `uv run skulk` landed on the default namespace, so a
  serviced node and a manual node on the same network could silently fail to
  form a cluster. The template now leaves the namespace unset (the shipped
  default for every launch path) and documents it as the opt-in isolation
  knob for running multiple Skulk clusters on one network. Existing installs
  keep their env files; only fresh installs see the new template.

- **Speech translation is now a standard capability.** `POST
  /v1/audio/translations` no longer requires `SKULK_ENABLE_EXPERIMENTAL_MODE`
  or the `experiments.speech_translation` config flag; like every other speech
  endpoint, its only gates are model truth (a mounted card declaring
  `audio.supports_translation = true`) and instance availability. With this
  graduation no built-in experiment remains active: the entire `experiments`
  config section (`tts_streaming`, `stt_realtime`, `speech_translation`) is
  deprecated accepted-but-ignored compatibility surface, and the dashboard no
  longer renders an Experiments settings section. The experimental-mode gate
  machinery stays in place for future features.

- **Fresh installs now use the same Zenoh data plane as the E2E qualification
  fleet.** An unset `SKULK_ZENOH_DATA_PLANE` selects Zenoh instead of silently
  falling back to gossipsub. Zero-config startup binds a specific
  private-LAN or CGNAT fabric IPv4 (loopback when offline or public-only) and
  enables local multicast scouting; public listeners require an explicit
  `SKULK_ZENOH_LISTEN`. An explicit `SKULK_ZENOH_CONNECT` list retains the
  multicast-off routed-fleet posture. `SKULK_ZENOH_DATA_PLANE=0` remains the
  explicit compatibility fallback. The native bindings version advances so
  service updates rebuild the new discovery-aware Zenoh constructor before
  startup.

### Added

- **Stale machine-generated model cards no longer shadow bundled cards.**
  Cards generated by `POST /models/add` now carry a `generator_revision`
  stamp. At load, a stamped custom card older than the running generator is
  superseded by the bundled card for the same model id with a loud warning
  (a generated card is cached metadata plus generator logic, not operator
  intent; a stale one silently pinned models to outdated engine selection).
  Hand-authored cards, which carry no stamp, keep full override precedence,
  and stale generated cards with no bundled counterpart still serve with a
  regenerate suggestion.

- **A Zenoh-isolated node is now named, not silently broken.** Every node
  advertises `zenohConnectedPeers` on `nodeResources`: the live peer-transport
  count of its Zenoh data-plane session, sampled at each advertisement behind
  a startup grace window so normal mesh formation never trips it. A node
  advertising Zenoh with a trustworthy count of 0 while other live nodes run
  Zenoh receives the error-level `zenoh_isolated` health reason in `GET
  /state` (dashboard badge included) and logs a recurring local warning with
  the fix, closing the shape where a member that multicast scouting cannot
  reach (for example one joined over a routed or overlay network) looks
  healthy while every remote stream through it dies with transport errors.
  The native bindings version advances so service updates rebuild the
  peer-count introspection before startup.

- **Laguna S 2.1 on the llama.cpp engines.** The managed llama-server pin
  advances to b10092, whose window landed the Laguna 2 model family and the
  native DFlash speculative arc upstream. The served engine gains the
  `draft_dflash` speculative type for models upstream's DFlash drafter arch
  supports. A bundled card serves Poolside's official Laguna S 2.1 Q4_K_M
  (one 128 GB unified-memory node, or pooled across several llama-server
  nodes via RPC); the card ships plain decode because upstream's DFlash
  drafter arch does not yet implement Laguna's gated attention, so no
  available drafter artifact can load (#676 tracks re-enabling).

- **Custom GGUF cards use both llama.cpp engines.** Cards generated by
  `POST /models/add` for a GGUF repo previously stamped only the in-process
  `llama_cpp` backends, so they never used a node's llama-server (losing
  its concurrency slots) and were silently ineligible for every multi-node
  GGUF placement, since only the served engine pools nodes via RPC.
  Generated cards now mirror the bundled GGUF cards: both engines
  compatible, served tags preferred.

- **Remote members join the fabric as first-class nodes.** A node whose
  advertised addresses are unreachable from its peers (a NAT'd or proxied
  cloud container reachable only through the connection it dialed in on) is
  no longer a floating, unplaceable entry in the topology. Every node now
  records its live, authenticated fabric connections as topology edges in
  their own right, annotated `session: true` in `GET /state`; placement can
  select such a member while host selection never mistakes the session's
  observed endpoint for a dialable address. Advertised addresses that keep
  failing their reachability probe drop to a slower retry cadence instead of
  being probed every sweep, so a remote membership no longer floods logs
  probing paths that can never work (#662).

- **Nodes that cannot reach the model store download directly from Hugging
  Face.** Store staging previously assumed every member could reach the
  store host; a remote node outside the store's network starved with a
  placement it could never fill. The availability probe now distinguishes a
  store that answered from a store that is unreachable at the transport
  level (including persistent mid-transfer dropouts), and an unreachable
  store routes the download to the model's origin on Hugging Face with the
  card's pinned revision preserved, logged loudly so a misrouted LAN node is
  still noticed. A reachable store answering with an error remains a store
  failure and never silently bypasses the store as the source of truth
  (#657).

- **Bare-install NVIDIA nodes complete the CUDA engine lane on demand.** A
  GPU-cloud container or plain checkout that never ran the installer's
  engine step previously degraded to a CPU-tagged engine while the hardware
  probe plainly saw the GPU. Provisioning now installs the pinned Foxlight
  CUDA engine wheel on demand (gated on the wheel's compiled compute-
  capability floor, with resolution pinned to the Foxlight and PyPI indexes
  and immune to host-level index overrides), verifies the installed wheel
  before claiming success, and degrades to the Vulkan/tarball chain with a
  copy-paste remediation when anything fails (#661).

- **Wire-version discipline makes incompatible builds fail loudly at
  connect.** The networking layer's private-network key now always derives
  from a `NETWORK_VERSION` constant (with the optional cluster namespace
  layered on top), so two builds whose wire protocols differ refuse to
  connect instead of half-working as a node that syncs events yet never
  appears in membership. Every wire-surface change must bump the version or
  record a wire-neutral judgment in `rust/networking/WIRE_COMPAT.md`,
  enforced by CI. The service startup script now rebuilds the Rust bindings
  whenever a pulled commit touches the Rust tree or workspace manifests and
  re-executes itself after a self-update, so an auto-updating node cannot
  keep running stale wire code while reporting itself current (#659).

- **Telemetry diagnostics count publishes that reached nobody.**
  `GET /v1/diagnostics/telemetry` now reports `noPeerPublishes` (publishes
  that found no peers subscribed on the telemetry protocol) separately from
  transport-pressure failures, and a node with live fabric connections whose
  telemetry has sustainedly reached nobody logs a rate-limited warning
  naming the consequence: the node will not appear in membership. A lone
  node, where no-peer outcomes are the normal state, stays quiet (#660).

- **The speech fabric: text-to-speech, transcription, and realtime voice.**
  Mounted TTS models serve OpenAI-compatible `POST /v1/audio/speech`
  (including streamed MP3/PCM for cards with proven streaming support,
  static voice catalogs via `GET /v1/audio/voices`, and bounded multipart
  reference audio for supporting cards); mounted STT models serve
  `POST /v1/audio/transcriptions` (with typed SSE or progressive NDJSON
  streaming where a card proves it) and standard
  `POST /v1/audio/translations`. A realtime transcription provider pins a
  session to a mounted speech worker and feeds a true upstream streaming
  session over bounded ingress; `WS /v1/realtime` is the OpenAI-compatible
  multi-turn adapter over it (24 kHz PCM16, optional server VAD with
  barge-in, and an optional response pipeline through a mounted chat model
  and TTS voice), and `WS /v1/fabric/chains/speech` exposes the same bridge
  as an explicit speech-to-chat-to-speech composition surface. A built-in
  WebRTC VAD provider emits typed turn boundaries on every production API
  node. The dashboard gains a full voice loop: assistant speech playback
  and microphone capture that uses realtime transcription when the mounted
  card supports it. Speech input and output ride dedicated node-addressed
  data paths, never the event log.

- **The vLLM served engine: the GPU concurrency fast path.** The worker can
  launch an external `vllm serve` process and proxy its OpenAI HTTP API as
  a second served-backend engine. Continuous batching and paged attention
  hold latency flat under concurrent load where single-stream engines
  collapse; the engine coexists with the llama.cpp engines and placement
  picks per hardware and expected concurrency. Single-node text generation
  in this first slice; enable per node with `SKULK_VLLM_BIN` or the
  installer's `--with-vllm`.

- **Concurrent serving on llama-server, with dynamic context.** The served
  llama.cpp engine now dispatches requests concurrently (`--parallel`
  slots behind a shared bounded-dispatch mixin), and the serving context is
  sized dynamically from placement-time memory fit instead of a fixed
  ceiling: discrete-VRAM nodes lift to the card's maximum where it fits,
  while nodes without discrete VRAM keep the conservative floor. GGUF model
  cards now prefer the served engine so concurrent serving is the default
  GGUF path.

- **Performance envelopes (observe-only).** The API node records one
  observation per completed generation into a bounded in-memory registry
  keyed by hardware, model, engine, and quantization, bucketed by in-flight
  concurrency at admission: p50/p90 time-to-first-token, decode rate,
  aggregate throughput, and a knee estimate. Exposed at
  `GET /v1/diagnostics/performance-envelopes` (with a cluster fan-out) and
  the dashboard Performance tab; deliberately off the event log and
  telemetry gossip.

- **Generated output rides an explicit stream lifecycle.** `DATA`-plane
  frames now carry `started -> chunk* -> completed|failed|cancelled` with
  per-command sequencing: the API orders and deduplicates frames, converts
  unresolved gaps into terminal transport errors with producer
  cancellation, and remote egress uses bounded independent per-command
  workers so one slow consumer cannot stall another stream. Extension
  provider media has the same contract on its own `PROVIDER_DATA` family,
  including client-streaming and bidirectional providers with caller
  half-close.

- **Model search across Hugging Face from the dashboard.** The model-store
  search can look up repositories and files directly on Hugging Face,
  so adding a model no longer requires leaving the dashboard to find the
  artifact.

- **Node Facts, derived capability, doctor, engine provisioning, and a
  one-command installer (the "skulk just works" program, #614).** Detection
  now creates serving capability and configuration overrides it, with every
  disagreement loud. One probe pass per process gathers a typed record of
  observed hardware (all GPUs, all vendors), observed software (importable
  dependencies, engine binaries and what they can drive via
  `llama-server --list-devices`), and declared `SKULK_*` configuration;
  backend derivation consumes it and advertises capability conflicts on node
  telemetry, `nodeHealth`, and the dashboard. A GPU node without backend env
  no longer serves silently on CPU (#609); an NVIDIA node missing
  `nvidia-ml-py` is loudly degraded and the binding is now a hard Linux
  dependency (#612); an invalid engine-binary override is named instead of
  read as unset (#462). `skulk doctor` runs the environment contract on
  demand with consequence-stating verdicts and `--fix` remediation, and its
  documentation is generated from the check registry. On Linux, Skulk
  provisions a pinned, checksum-verified upstream llama-server build on
  demand (`SKULK_LLAMA_SERVER_BIN` still overrides;
  `SKULK_NO_ENGINE_AUTOPROVISION=1` opts out), and `install.sh` takes a
  fresh macOS or Linux box to a working node in one command. On NVIDIA
  Linux GPU nodes, the preferred managed engine source is a pip-installable
  wheel built from pinned upstream source in Skulk's own CI:
  `skulk-llama-server-cuda` (NVIDIA; CUDA runtime from NVIDIA's official
  PyPI packages) and `skulk-llama-server-vulkan` (AMD; Khronos loader
  bundled), both carrying sigstore build provenance and published to the
  Foxlight wheel index at `wheels.foxlight.ai` (the authoritative source;
  the Vulkan wheel is additionally mirrored to PyPI), with a workflow guard
  keeping the engine pin, wheel versions, and installer in lockstep.

- **Explicit, auditable cluster heartbeat.** Nodes now publish a dedicated
  telemetry heartbeat instead of making liveness an accidental side effect of
  collector cadence. The master warns before the prune window, retains ordinary
  telemetry and control events as fallbacks, and records every deciding signal
  age in `NodeTimedOut` so a removal remains explainable after replay (#448).

- **Telemetry can no longer congest correctness-critical control traffic.**
  Local producers enter a bounded latest-value admission map and telemetry uses
  its own Python egress loop plus a dedicated gossipsub protocol and per-peer
  handler queues. Intermediate download progress now rides this lossy plane;
  only completed and failed outcomes remain in event-sourced `State`, with
  attempt identities preventing a late progress sample from overriding a
  terminal or reset decision. `GET /v1/diagnostics/telemetry` reports aggregate
  admission, coalescing, drop, queue, failure, byte, and age metrics (#565).

- **Placement previews expose every valid host, not just the ranking
  winner.** `GET /instance/previews` now includes per-host single-node
  previews marked `alternative: true` for each host that passes admission
  but lost the planner ranking, and the dashboard placement dialog derives
  node eligibility from the planner's answers instead of a chip-family
  heuristic. Previously a heterogeneous fleet showed only the ranked pick
  (typically the largest free GPU) as placeable, and the heuristic rendered
  a CUDA node's pill as unable to run GGUF while the planner was placing
  GGUF on it (#557).

- **Nodes can name themselves, and CUDA devices get their own topology tile.**
  `SKULK_NODE_NAME` overrides the gossiped display name ahead of the
  hostname/Computer Name fallback, so containers and rented GPU pods (whose
  runtime-random hostnames cannot be changed without privileges) identify
  themselves properly in the dashboard. Nodes whose telemetry reports an
  NVIDIA accelerator now render as a spark-style CUDA device tile with the
  NVIDIA wordmark, in the same visual family as the Mac and AMD tiles,
  instead of the generic hexagon (#555).

- Prebaked CUDA pod image (`deployment/cuda/Dockerfile`, published to GHCR as
  `skulk-cuda-pod` by the `cuda-image` workflow): carries the CUDA
  llama-cpp-python wheel, `llama-server` + `ggml-rpc-server` binaries, uv,
  and the Rust toolchain, so a rented GPU pod goes from create to serving in
  minutes (`/opt/skulk/pod-bootstrap.sh <ref>`) instead of the ~1 hour
  install recipe. The recipe (`install-deps.sh`) remains the from-scratch
  path for arbitrary driver-equipped machines.

- NVIDIA / CUDA node support (platform plumbing): a passive NVML telemetry
  collector (`utils/info_gatherer/nvidia_gpu.py`) fills the normalized
  accelerator profile on NVIDIA nodes (the Linux GPU monitor tries AMD
  sysfs first, then NVML), and `deployment/cuda/install-deps.sh` provisions
  a driver-equipped machine (e.g. a rented GPU pod) into a serving node
  with the CUDA llama-cpp-python build and optional CUDA `llama-server`.
  Backend advertisement reuses the existing `SKULK_LLAMA_CPP_BACKENDS=cuda`
  declaration with build cross-checking.

- Opt-in field telemetry (off by default): a first-run dashboard consent
  modal and permanent Settings toggles control anonymous performance and
  reliability samples (model id, hardware class, timing, token counts,
  failure classes; never prompts, outputs, or machine identity). Consent
  persists in `skulk.yaml`; `GET /v1/telemetry/preview` shows the exact
  pending batch; `SKULK_TELEMETRY_DISABLE=1` hard-disables per node.

- **Extensions can call capabilities (the generic call verb).** The unary loop
  of the provider surface is complete: a provider extension that implements
  `handle_call` becomes callable, and any extension invokes a discovered
  capability with `ExtensionContext.call_capability(node, id, version,
  revision, payload)`. Calls are node-addressed and direct (the master is never
  in the hot path; nothing is event-sourced), pin the exact `id@version` plus
  the descriptor revision digest from discovery so a drifted contract is
  rejected instead of misinterpreted, and are schema-validated in both
  directions (bounded JSON Schema 2020-12 validation that never fetches remote
  references). Every failure is a typed, machine-readable error code on the
  result rather than an exception. Calls are bounded: a deadline (default
  30s), 1 MiB payload and result caps, and a per-node concurrency bound that
  rejects excess calls as `overloaded` instead of queueing them. Served over
  the new `POST /v1/capabilities/call` endpoint; calling the local node is an
  in-process fast path with identical guards. The reference echo provider now
  serves calls end to end.

- **Extensions can serve self-describing capabilities (the provider role).**
  Skulk cannot enumerate future plugin capabilities, so it standardizes the
  description instead: a provider extension publishes one `CapabilityDescriptor`
  per capability it serves, carrying the capability id, a semantic version, a
  human/LLM-readable description, JSON Schemas for input and output, the call's
  I/O mode (`unary`, `server_streaming`, `client_streaming`, `bidirectional`),
  and a content revision digest that detects any drift in the published shape.
  Discovery is two-layered: the descriptor's id is auto-advertised as the node's
  telemetry capability tag (cheap, gossiped), and the full descriptors travel on
  demand via `ExtensionContext.describe_node(node_id)` or the new
  `GET /v1/capabilities` endpoint (heavy, fetched). Providers get an `on_start`
  startup hook (a pure provider has no chat hook through which to reach the
  context), and `withdraw_capability(tag)` reverses an advertisement, with the
  telemetry publisher emitting one final empty reading when a node's last tag is
  withdrawn so peers clear their entry instead of holding a stale value. A
  reference provider lives at `examples/extensions/echo-provider/`.

- **Extensions get telemetry-plane access (read + advertise).** First-class
  fabric citizenship expressed as plane access: a plugin can now both discover
  the cluster it belongs to and announce what it offers.
  - `ExtensionContext.read_cluster()` returns an immutable, per-node snapshot of
    the telemetry plane: each node's backends, participation role, accelerator
    vendor, Skulk version, RAM, liveness, and any capability tags peers
    advertise. The call is a pure in-memory snapshot (no network I/O, no
    mutation), and every field is `None`/empty until that reading has arrived.
  - `ExtensionContext.advertise_capability(tag)` publishes an opaque capability
    tag (for example `"memory"`) onto the telemetry plane so peers discover it
    the same way native nodes advertise their backends. The tag surfaces in
    every peer's `read_cluster()` snapshot under `ClusterNodeView.capabilities`.
    Advertising is additive and idempotent; the tag keeps being gossiped
    (last-write-wins) until the node leaves the cluster.

### Fixed

- **Topology nodes minted by an edge alone are reaped with their last
  edge.** A fabric connection can be the first the cluster hears of a peer,
  creating its topology entry before the peer publishes any node
  information. A peer that disconnected without ever becoming a member left
  that entry behind forever, because the membership timeout only reaps nodes
  it has heard from; a crash-looping box could litter the graph with a new
  phantom entry per restart attempt. Deleting the last edge pointing at a
  never-a-member node now removes the node itself, cascading through any
  dangling edges the dead peer emitted that nobody remains to delete; real
  members are untouched. Session edges are additionally emitted only for
  current members, and each node re-emits its live member edges if state
  loses them, so a timed-out peer's lingering socket cannot re-mint a
  phantom while a recovering peer's edge returns within one sweep of its
  membership republication (#671).

- **The dashboard node card shows VRAM for discrete-GPU nodes.** The card's
  memory figure treated any reported VRAM as a unified-memory carve-out and
  added it to system RAM, so a discrete-GPU node on a big-memory host
  displayed host RAM plus VRAM (a 45GB A40 on a 512GB cloud host read
  548.5GB) when VRAM governs what the node can serve. Discrete GPUs now show
  their VRAM pool used/total with an explicit label, classified by the same
  GTT-aperture signature placement uses, so discrete AMD cards are handled
  correctly alongside NVIDIA; unified-memory nodes (Apple, AMD APU
  carve-out) are unchanged. Placement admission was never affected (#669).

- **Automatic repair re-placements honor the operator's node exclusions.** A
  placement created with `excluded_nodes` stamps those exclusions onto the
  instance, so when a node dies and the master re-places the instance, the
  repair search still avoids the operator's excluded nodes instead of
  silently forgetting them (#658).

- **A served-MTP model with a missing speculative draft serves without
  speculation instead of crashing.** When a served card declares a
  cross-repo draft (`served_spec_draft_repo`/`served_spec_draft_file`) whose
  GGUF is not on disk, the `llama_server` runner used to raise `FileNotFoundError`
  at launch and crash the instance. The draft is a best-effort companion (a
  failed cross-repo co-fetch is swallowed and the base is still marked
  complete), so a declared-but-absent draft now degrades to plain decode: the
  runner drops `--spec-type`/`--model-draft` and logs a warning rather than
  failing a loadable model (#574, sharpens #554).

- **The CUDA pod entrypoint wires an injected `HF_TOKEN`.** A Hugging Face
  token passed as the `HF_TOKEN` pod environment variable (same pattern as
  `PUBLIC_KEY`) is now persisted to the canonical Hugging Face token file so
  skulk's downloader authenticates every model fetch. Without it, downloads
  from gated or Xet-backed repos fail and a `--ensure-store-downloads` run
  stalls on a download that never starts (#575). The prebaked image must be
  rebuilt to pick up the new entrypoint.

- **Multi-homed peers no longer churn connections on every weak ping or dead
  link-local retry (#401).** mDNS still tries every advertised path once so a
  reachable Thunderbolt link is retained, but once another path connects,
  failed link-local addresses retry once per minute instead of every five
  seconds. A socket now requires three consecutive five-second ping failures
  before teardown. Routable paths, API reachability discovery, and real peer
  loss retain their normal behavior.

- **Retained event-log replay no longer arrives as an unpaced 10k-event
  burst.** The master coalesces replay requests onto one background worker,
  emits the retained tail in bounded paced chunks without blocking command
  processing, and warns when the event log grows at an elevated rate while
  the cluster has no active task or download. This removes the replay
  amplifier that could turn slow periodic event growth into follower flaps
  and repeated state-sync storms (#449).

- **Realtime Fabric speech replies cannot generate indefinitely before TTS.**
  Automatic chat responses now enforce a configurable 1-4096 output-token
  ceiling (256 by default) and disable hidden reasoning unless explicitly
  requested, so a model that does not emit EOS or spends its budget reasoning
  cannot consume the entire WebSocket deadline or leave the selected speech
  participant without visible text.

- **Abandoned Zenoh DATA streams no longer retain admission forever.** Each
  remote command queue has a renewed-on-frame 30-minute resource lease. An
  omitted terminal now tombstones and closes the stream, releases owner and
  process admission, best-effort emits a correctly sequenced typed failure, and
  increments global/per-owner `idleStreamReclaims` diagnostics instead of
  leaving an empty active queue until process restart (#567).

- Control-plane saturation can no longer starve master election: election messages
  use dedicated Python egress plus an isolated libp2p gossipsub protocol and
  handler queue, while a deduplicated legacy copy preserves rolling upgrades.
  Repository download callbacks are bounded, coalesced, terminal-ordered, and
  aggregate-only on the event path; queue-pressure logs are rate-limited and
  omit failed payloads.

- **GPU llama.cpp nodes self-heal a pruned inference wheel at startup.** A node
  that declares a GPU llama.cpp backend builds its `llama-cpp-python` wheel
  from source; `uv sync --inexact` preserves a present wheel, but could not
  restore one that a plain `uv sync` (run by hand or another tool) had pruned,
  leaving the node silently unable to import `llama_cpp` and dropping it out of
  all GGUF and served-MTP placement with no error. The startup script now
  verifies the wheel after sync on a GPU node and rebuilds it from source once
  (Vulkan for AMD, CUDA for NVIDIA) if it is missing or CPU-only. Non-fatal and
  single-shot: the node still serves without the in-process GGUF engine if the
  rebuild fails, and the common case (wheel present) skips it (#568).

- **Cluster observability answers fast and accounts for every node.** The
  diagnostics fan-out probed each peer address with a patient retry policy
  meant for targeted lookups, so a single unroutable advertised address
  (an overlay-joined node, a docker bridge, a dead interface) stalled the
  whole observability surface for ~18 seconds; fleet-wide sweeps now use a
  fail-fast single-attempt policy (measured 18.4s -> 3.3s on a six-node
  fleet with unroutable candidates). Peers with no reachable API route now
  appear in the response as explicit failures instead of silently vanishing,
  so a node that joined over an overlay keeps an observability presence
  (#558).

- Text-generation compatibility endpoints now reject mounted TTS-only and
  STT-only model cards before command dispatch, preventing modality mistakes
  from reaching or restarting speech runners.

- **llama.cpp engines now report generation statistics.** Chat completions
  served by the in-process `llama_cpp` engine and the served `llama_server`
  proxy attach real `generation_stats` (prompt/generation token counts and
  tokens-per-second) to the terminal chunk, as the MLX engine always has.
  `llama_server` uses the server's own engine-side timings when available
  (requested via `timings_per_token`), falling back to proxy-side wall-clock
  phases; the in-process engine measures prefill/decode phases around its
  stream. Previously every request from these engines carried
  `stats=None`, leaving the dashboard, field telemetry, and harness
  `skulk_*_tps` metrics blind on GPU/Linux nodes (#532).

- **Store-host staging no longer doubles disk usage.** Staging a model from a
  local store into the worker staging directory hardlinks each file instead
  of copying when both live on the same filesystem (store files are immutable
  once registered and staged files are never mutated in place), falling back
  to a copy across filesystems. Previously a 26GB GGUF needed 52GB of free
  disk to stage on a store-host node, and the staging copy could fail with
  ENOSPC after a successful store download (#533).

- **Pooled (multi-node) GGUF placement no longer caps unified-memory nodes at
  their BIOS VRAM carve.** Admission for RPC placements previously sized each
  node against a VRAM-carve-only figure, which falsely refused pooled models
  on AMD Strix Halo nodes: the carve is not an allocation boundary on a
  unified-memory APU (the GPU maps system RAM through GTT), and the figure
  was further deflated by transient `vram_used` readings. Pooled admission
  now uses the same UMA-aware usable-GPU figure as single-node placements.
  Proven live on a Strix Halo pair: a pooled gpt-oss-120b placement,
  previously refused with the donor capped at "17.7GB (usable GPU VRAM)",
  loaded 40.6GB on the driver and 22.8GB on the donor and served coherently
  at 44 tok/s decode. A node in an RPC cycle whose accelerator telemetry has
  not arrived still surfaces as info-pending rather than falling back to the
  system-RAM formula.

## [1.4.1] - 2026-07-06

### Added

- **The dashboard works from a phone.** At phone widths (480px and below)
  every view adapts to a single-column layout: the header collapses into an
  animated hamburger menu carrying the navigation, observability, settings,
  and theme controls; the model store table becomes stacked cards; the
  conversation-history and active-instances panels open as drawers over the
  content (one at a time, fully opaque, dismissed by tapping the dimmed
  backdrop); observability takes over the full screen and sizes itself to the
  real visible area on iOS Safari; the cluster topology scales its node cards
  and thins the background mesh for the smaller canvas; and the chat input
  reflows with the model selector on its own line. Verified with headless
  sweeps at 360px, 390px, and 414px showing no horizontal overflow on any
  view. A new documentation page, "Manage Your Cluster from a Phone"
  (`website/docs/mobile-dashboard.md`), covers the mobile layout and
  remote access, with Tailscale as the recommended remote-access shape.

### Fixed

- **The chat scroll-to-bottom button works.** In any fresh session, the
  scroll-position restore logic re-armed itself off the user's own scrolling
  and yanked the pane back, cancelling in-flight smooth scrolls; the button
  visibly did nothing. It also passed its click event where a scroll behavior
  belongs, and stacked above the mobile drawers instead of beneath them. All
  three fixed; smooth scrolling in chat is reliable again generally.

- **The observability Node tab reports each node's own Tailscale state.** It
  previously queried the dashboard-serving node's local Tailscale status and
  displayed it for every node in the cluster. The reading now rides each
  node's diagnostics bundle. The Tailscale probe is TTL-cached on the API
  node, and a probe child that outlives its timeout is killed and reaped
  instead of leaking one stuck subprocess per poll. A spurious "current
  master is not a placement node" warning that fired on normal topologies
  was removed.

- **GGUF engines now report runner phases to observability.** The `llama_cpp`,
  `llama_server`, and RPC-donor runners emitted no runner-phase diagnostics at
  all, so the dashboard's observability Live tab sat at "created" forever for
  any placement on those engines (every GGUF placement on a GPU/Linux node).
  All three now record the same lifecycle the MLX runner does: task
  acknowledgement, model load (server spawn for the served and donor shapes),
  ready, generation start, completion or cancellation, errors including a
  server subprocess dying behind the runner, and shutdown teardown.

- **The dashboard header shows the real Skulk version.** The UI version was
  baked at build time from the dashboard's own `package.json`, which had
  silently drifted (it still said 1.3.0 through two releases). The build now
  reads the version from the repo root `pyproject.toml`, the single source
  of truth the release process bumps, so the UI can never drift again. The
  dashboard `package.json` version is no longer read by anything.

## [1.4.0] - 2026-07-05

### Added

- **Multi-node GGUF inference: memory pooling across GPU nodes.** A GGUF model
  that fits no single GPU node but fits the combined GPU memory of several
  `llama_server` nodes now places and serves as a driver-plus-donors pair: one
  driver node runs `llama-server --rpc donor:port,...` and holds the model
  file, each donor runs a small `ggml-rpc-server` that lends its GPU memory,
  and llama.cpp splits the weights and KV across the pooled devices itself.
  Placement admits pooled models against each node's VRAM carve (measured:
  RPC allocations never use the Strix UMA/GTT spill), picks the biggest-VRAM
  node as the driver, and stamps routable donor endpoints chosen from the
  observed connectivity (preferring a USB4/Thunderbolt link when present;
  link-local addresses are rejected). Single-node placement is always
  preferred whenever the model fits one node. The multi-node cycle rule also
  fixes a latent placement hole where a card compatible with several engines
  could admit a cycle mixing nodes that cannot form one ring (#414).
  Text-generation GGUF model cards gained `llama_server` compatibility so the
  catalog can pool; per-card backend preferences are unchanged pending
  in-process vs served single-node measurements. New env var:
  `SKULK_RPC_SERVER_BIN` (optional; defaults to the `ggml-rpc-server` next to
  `SKULK_LLAMA_SERVER_BIN`). (#328)

- **Cards declare model truth; platform limitations moved to code.** A model
  card's `compatible_backends` now records only which engines the model's
  artifacts run on. Capabilities our runners cannot yet exploit (currently:
  the served llama.cpp engine cannot load a vision model's projector) are
  gated in a code-level capability table (`platform_compatible_backends`)
  applied by placement and worker engine resolution, so a vision model never
  lands where its advertised capability would silently degrade, and cards
  need no edits when the platform catches up. Backend fallback resolution
  now orders CPU compute tags after GPU tags when a card expresses no
  explicit preference.

- **Extension (plugin) API.** Skulk now discovers separately installed Python
  packages through the `skulk.extensions` entry-point group at startup and
  calls them at well-defined serving-path hooks: a chat-request transform
  before cluster dispatch and a completed-response observer after streaming
  ends, with an `ExtensionContext` giving in-process access to the cluster's
  embedding serving. Extension calls are guarded (a raising extension is
  logged and skipped rather than failing the request), extensions never own the
  response stream (Skulk accumulates and hands observers an immutable
  summary), and version gating refuses plugins whose `skulk_requires`
  specifier does not match the running Skulk. `SKULK_EXTENSIONS_DISABLE=1`
  is a node-local kill switch. No extension installed = Skulk unchanged.

### Fixed

- **Joining a cluster no longer triggers a connectivity-gossip storm.** Workers
  previously re-emitted their connectivity readings on every gather tick, so a
  long-lived cluster accumulated a huge replay tail that a joining node had to
  ingest all at once, saturating send queues and flapping nodes (worst on AMD
  nodes joining a mostly-Mac fleet). Connectivity events are now emitted only
  when the readings actually change, and node liveness (pruning plus the
  dashboard health indicator) rides telemetry freshness instead of event-log
  heartbeats. (#447)
- **Linux network interfaces are now typed.** Interface classification
  previously worked only on macOS, so every Linux NIC reported `unknown` and
  the Thunderbolt-first address prioritiser could never fire between two Linux
  nodes. Linux interfaces are now classified via sysfs (thunderbolt, ethernet,
  wifi), so a USB4/Thunderbolt link between GPU nodes is preferred
  automatically. (#450)
- **Placement previews report the instance shape that would actually serve.**
  A preview used to echo the requested instance meta even when placement would
  mint a different shape; the preview now derives its meta from the minted
  instance itself. (#452)
- **A node's failed download no longer poisons future placements.** Recovery
  from a terminally failed model download now resets that node's download
  record; previously the stale failure lingered in session state and condemned
  every later placement of the same model touching that node (observed live:
  one out-of-disk error kept killing fresh placements long after space was
  freed, until a whole-fleet restart). (#454)
- **A worker refusing a placement now falls back instead of giving up on
  heterogeneous clusters.** When a node refuses its shard (for example the
  memory fit guard) and no wider cycle exists, the master now retries the
  model at single-node width excluding the refuser, instead of tearing the
  placement down permanently. A refusal against that fallback is terminal
  (bounded at two hops, never an oscillation), and terminal teardown also
  cancels the model downloads the doomed placement started. (#455, #456)
- **Cluster listener ports moved out of the OS ephemeral range.** Ring,
  coordinator, and RPC donor ports are now allocated from a reserved band
  below both the Linux and macOS ephemeral floors and exclude ports already
  held by live instances, eliminating bind collisions with short-lived OS
  connections; port exhaustion now fails placement loudly instead of looping.
  (#457)
- **A served engine or RPC donor process that dies between requests is now
  detected.** The worker polls subprocess liveness between tasks, so a
  crashed `llama-server` or `ggml-rpc-server` marks the runner failed (with
  the subprocess log tail in the error) instead of leaving a Ready runner
  over a zombie that wedges the next request. (#451)
- **Bundled model cards audited end to end; every finding fixed and gated.**
  A full audit of the 136 bundled cards (schema, cross-field invariants,
  capability resolution, and a live check of all 148 referenced Hugging Face
  repos and file paths) found and fixed: the Ornith 1.0-35B MLX card pointed
  at a deleted repo and now uses the official `mlx-community` conversion
  (structurally identical, values re-derived from the new artifacts); the
  Qwen3 family capability default forced a thinking contract onto
  instruct-only variants that explicitly declare no thinking (five bundled
  Instruct-2507 / Next-Instruct cards resolved wrong; the resolver now
  respects explicit card capabilities while keeping the auto-imported-card
  default); seven cards (DeepSeek V3.1/V3.2, gemma-3n, gemma-4-e4b) were
  missing `context_length` and three Nemotron-3-Nano cards were missing
  `num_key_value_heads`, all filled from each repo's real config; and two
  GLM-5 card filenames did not match their model IDs. Every static audit
  invariant now runs as a per-card test gate, so a bad bundled card fails CI
  instead of shipping.

## [1.3.1] - 2026-07-01

### Added

- **Complete, capability-accurate model cards for the whole store.** Every model in
  the store now has a card, and every committed card is complete against the current
  `ModelCard` schema with each property derived from the model's real capabilities
  (structural fields from the HF `config.json` via `fetch_from_hf`, capability fields
  from the HF model card). Added cards for previously-uncarded store models
  (Qwen3-4B / Qwen3-4B-Instruct-2507 / Qwen3.6-35B-A3B, Devstral-Small-2-24B,
  Moonlight-16B-A3B, and the GGUF serving models gpt-oss-20b/120b, Qwen2.5-7B,
  Llama-3.3-70B, Qwen3-Coder-30B, gemma-4-31B, Llama-3.2-1B, Qwen2-VL-2B). Audited
  and corrected the existing cards, including capability fixes grounded in the real
  models: Step-3.5-Flash is always-reasoning (no thinking toggle) and several Qwen
  VLMs were missing their vision section. Cards advertise only what the serving
  engine can actually deliver, so served-MTP GGUF cards stay text-only (the
  llama_server engine has no vision projector) and the gemma-4 GGUF card keeps its
  reasoning even though the in-process llama_cpp path does not yet split it.

- **Dashboard renders AMD Ryzen AI Max nodes as their own device.** The topology
  graph and cluster cards now draw a dedicated AMD Strix Halo glyph (detected from
  the SoC/chip string) instead of a generic node or a Mac, and AMD APU nodes report
  their full unified memory (the VRAM carve-out plus system RAM) the same way Macs
  report unified memory, rather than only the system-RAM slice.
- **Running-instance cards show every placement node and its status.** A multi-node
  instance card now lists all of its nodes with per-node state (ready, loading,
  failed, and so on) instead of a single node, so a lagging node is obvious during
  load.
- **`GET /v1/models` reports a model's speculative-decoding companions.** Each
  entry's `runtime` section now includes `mtp_sidecar_repo`, `assistant_model_repo`,
  and `served_spec_draft_repo` when the card declares them, so clients can tell a
  placeable model apart from its drafter or MTP-head companion.

- **Store-delete now evicts worker-staged copies cluster-wide (#427).** Deleting a
  model from the store (`DELETE /store/models/{model_id}`) previously removed only
  the store host's canonical copy; workers cache their own staged shards
  independently, so the deleted model lingered on worker disk until LRU pressure.
  The API now broadcasts a fleet-wide `EvictStagedModel` command after a successful
  store-delete: every node drops its local staged copy and the model's download
  entries are cleared from cluster `State`, so the planner re-stages on a future
  placement instead of loading deleted files.

- **Served-backend engine (`llama_server`) with native MTP speculative decoding.**
  A new inference-engine class that launches an external `llama-server` subprocess
  and proxies its OpenAI HTTP API, coexisting with the in-process `mlx` and
  `llama_cpp` runners. This unlocks llama.cpp's native multi-token-prediction
  (`--spec-type draft-mtp`) for models that ship MTP heads (Qwen3.6, DeepSeek,
  GLM, Kimi, Nemotron), which is not reachable from the in-process Python binding.
  Routed per model via a card's `compatible_backends` and configured with the
  `served_spec_type` / `served_spec_n_max` runtime fields; enabled on a node by
  pointing `SKULK_LLAMA_SERVER_BIN` at a `llama-server` binary. Measured 2.19x on a
  dense Qwen3.6-27B on a Strix Halo (Radeon/Vulkan).

### Changed

- **The placement modal only offers options that apply to the model.** Networking
  (MLX Ring / Jaccl) is hidden for single-node GGUF models (the llama.cpp and served
  engines have no MLX transport to pick); the node selector (exclusion pills and the
  count slider) appears only when more than one node can host the model; and nodes
  that cannot run the model (wrong engine or hardware) are shown disabled instead of
  clickable.
- **The model store no longer presents drafters and MTP-head sidecars as launchable
  models.** Speculative-decoding companion repos (a separate draft model, or an MTP
  prediction-head sidecar) now carry a "Drafter" or "Sidecar" badge and have no
  launch, placement, or optimize actions, because they are downloaded and loaded
  automatically with their parent model. The OptiQ (mlx-optiq) optimize action is
  also hidden for GGUF models, since it only applies to MLX weights.
- **The bundled Qwen3.6-27B MTP card now points at `unsloth/Qwen3.6-27B-MTP-GGUF`**
  instead of a small community mirror that HuggingFace throttled to roughly 1 MB/s.
  The 17 GB weights previously never finished downloading into the store (they were
  re-attempted from near-scratch on every placement); the well-provisioned unsloth
  repo downloads and finalizes in a few minutes so the store copy persists and
  re-stages instantly. Same base model, same native multi-token-prediction heads,
  same `--spec-type draft-mtp` path.

### Fixed

- **Correct engine labels and device glyphs across the dashboard.** Served
  (llama.cpp / `llama-server`) instances were mislabeled "Pipeline / MLX Ring" and
  were missing their MTP badge on the instance card, the model-store card, and the
  store's ready-hover card. The engine is now derived from the model card's backends,
  so GGUF and served models read as "llama.cpp" and draw the correct AMD device glyph.

- **Large model downloads no longer time out mid-transfer.** The download session's
  `long` timeout profile applied a fixed 30-minute `total` cap to the entire file
  transfer, so a multi-GB GGUF that was downloading fine failed partway through
  once it outlasted the cap (a 17 GB model at ~7.5 MB/s hit the cap at ~80%,
  surfacing only as an empty-string `TimeoutError`; larger models like the 62 GB
  gpt-oss-120B GGUF would fail sooner in percentage terms). Large file-body
  downloads now have no `total` cap and are instead policed by `sock_read` /
  `sock_connect` inactivity timeouts: a genuinely stalled connection still times
  out and retries (resuming from the `.partial`), while a slow-but-alive transfer
  of any size completes. The worker-side wait for a store download
  (`request_and_wait_for_download`) is likewise now progress-aware: its timeout
  is a stall timeout (max time without progress), not a total cap, so the worker
  no longer gives up on a live, still-progressing multi-hour download and lets
  the master tear the placement down. Store download failures also now record the
  exception type instead of an empty error string.

- **Store re-download after a delete no longer silently no-ops.** `ModelStore`
  caches per-model download status in memory; `delete_model` removed the registry
  entry and on-disk files but left a stale `"complete"` status behind, so a later
  `request_download` short-circuited on it and never re-fetched. The model would
  then appear "complete" while absent from the registry and disk, and a worker
  staging it failed with "not found in store". `delete_model` now clears the
  cached status, and `request_download` treats a cached `"complete"` as stale
  whenever the model is no longer actually in the store (a backstop for any cause
  of files-gone, including out-of-band removal). This unblocks re-provisioning a
  model after a store-delete (e.g. the download/delete/re-download cycle the test
  harness drives for served-MTP GGUFs).

### Documentation

- **Comprehensive docs correctness and beginner-readability sweep.** Corrected the
  AMD/Strix Halo docs to state the inference backend is llama.cpp Vulkan (Mesa RADV),
  not ROCm (ROCm is optional, used only for the `rocminfo` diagnostic); fixed a
  fabricated native-MTP model list and a macOS-only log-path claim; added in-site
  install and first-run commands so a newcomer can reach a running node without
  leaving the docs; removed internal roadmap, PR, and incident lore from user-facing
  pages; and removed em dashes from the docs prose.

## [1.3.0] - 2026-06-25

This release makes Skulk a **heterogeneous** inference fabric: alongside Apple
Silicon (MLX), an AMD or other Linux GPU node can now join the same cluster and
serve GGUF models through a new llama.cpp engine (Vulkan / ROCm). It also lands
the control / telemetry / data plane separation (with an optional Eclipse Zenoh
data-plane transport), a centralized model store, vision GGUF support, and
cross-engine reasoning (gpt-oss harmony and `<think>` parsed into a separate
reasoning channel), plus a large amount of placement, failover, and stability
hardening. See `website/docs/release-notes/1.3.0.md` for the user-facing summary.

### Fixed

- **`<think>` reasoning no longer leaks into the answer for reasoning models on
  the llama.cpp engine.** A token-delimited reasoning model (e.g. Qwen3.5-MoE /
  Ornith) emits its chain-of-thought inside `<think>...</think>`; llama.cpp hands
  back already-detokenized text, so the whole reasoning block flooded the visible
  `content` (the answer buried after it) and nothing reached `reasoning_content`.
  The llama.cpp runner now reparses the marker stream from strings, mirroring what
  the MLX engine does at the token level: text inside the markers becomes
  `reasoning_content`, the answer becomes clean `content`, and the markers are
  stripped. This generalizes the gpt-oss harmony fix to plain `<think>` reasoning;
  the parser is dependency-free (no MLX) so it runs on non-Mac GPU nodes.

### Changed

- **The llama.cpp engine now loads models with Flash Attention by default.** It
  is the modern llama.cpp default and matters most for models whose per-layer V
  embeddings differ (gemma's interleaved sliding-window attention): without it
  llama.cpp pads the V cache and falls back to a full-size sliding-window cache,
  wasting VRAM and slowing attention. Set `SKULK_LLAMA_CPP_FLASH_ATTN=0` to
  disable on a backend whose compiled build lacks Flash Attention kernels.

### Fixed

- **The model store now advertises a routable IP, so downloads no longer fail
  on Thunderbolt-meshed fleets.** The store host broadcast its `store_http_host`
  as a bare hostname (e.g. `kite3.local`); on a fleet where nodes are also linked
  over Thunderbolt, mDNS could resolve that name to the host's link-local TB
  address (`169.254.x`), which a peer lacking a direct TB link to it cannot route
  to. That peer's model downloads then failed (`Cannot connect to host
  kite3.local:58080`) even though it could reach the store fine over the LAN. The
  store host now broadcasts its own best routable IPv4 (a private LAN address is
  preferred over a Tailscale/CGNAT address; loopback and link-local are skipped),
  and an operator-supplied routable IP in `store_http_host` is still honored
  verbatim.

- **gpt-oss conversations no longer wedge on follow-up turns when the history
  carries raw harmony markers.** A gpt-oss assistant turn captured before the
  output was channel-parsed (or echoed back by any client) keeps `<|channel|>`
  markers in its `content`, and llama.cpp's gpt-oss chat template rejects that
  with a hard error, so every later turn of the conversation returned no
  response. The llama.cpp runner now strips harmony markers from assistant
  history (reducing it to the final-channel text) before handing it to the
  template, so a client can never wedge inference by replaying the model's own
  output format and existing conversations resume cleanly.

- **Dashboard: gpt-oss chats render cleanly and stop leaking harmony markers.**
  The chat reasoning/content splitter now also understands gpt-oss "harmony"
  channel markers (`<|channel|>analysis...final...`), so the analysis channel
  shows as collapsible reasoning and the answer renders without raw `<|...|>`
  scaffolding. This also heals conversations that stored marker-laden assistant
  turns before the server-side parsing landed, so they display correctly.
- **Dashboard: no more `404 /i18n/skulk/en.json`.** English ships bundled in the
  app, so the Tolgee CDN fetch is now only enabled when an additional language is
  configured; the default English-only build no longer requests (and 404s on) the
  runtime translations endpoint.

- **gpt-oss models on the llama.cpp engine no longer leak raw "harmony" markers
  into the answer, and their reasoning is now separated from content.** llama.cpp
  hands back already-detokenized text whose content still contains the literal
  harmony channel scaffolding (`<|channel|>analysis<|message|>...<|end|>`
  `<|start|>assistant<|channel|>final<|message|>...`); the runner forwarded it
  verbatim, so users saw the control tokens and the analysis (reasoning) channel
  mixed into the response. The llama.cpp runner now reparses the marker stream
  from strings, mirroring what the MLX engine does at the token level: the
  `analysis` channel is emitted as `reasoning_content`, the `final` channel as
  clean `content`, and every control marker is stripped. The parser is
  dependency-free (no MLX / openai_harmony) so it runs on non-Mac GPU nodes.

- **Cluster formation no longer livelocks under connection churn (#400).**
  Master election restarted its whole campaign on every libp2p connection
  update, and because peers are multi-homed (link-local, LAN, and overlay
  addresses) and libp2p pings/re-dials every few seconds, those updates can
  arrive faster than the election timeout. A campaign was then cancelled and
  restarted before it could ever finish, so no master was elected and the
  cluster never formed (most visible when several nodes start at once, e.g. a
  fresh three-machine setup). Election now ignores connection updates that do
  not change the set of connected peers, and lets an in-flight campaign finish
  before starting the next one, so steady churn can no longer starve
  convergence. Reducing the churn at its source (skipping unreachable
  link-local dials, ping tuning) is tracked separately (#401).

- **A placement right after a teardown is no longer spuriously refused on
  gossip-lagged memory (#314).** Node memory rides the telemetry plane
  (last-write-wins), so for a few gossip rounds after a runner teardown the
  freed memory is not yet reflected in `ramAvailable` (or the GPU-wireable
  figure). A back-to-back placement (test harness, rapid model swap) reads the
  stale, deflated availability and is refused with "no candidate cycle fits"
  until a ~20-30s settle. The master now credits a just-deleted instance's
  per-node footprint (estimated with the same accounting placement admits
  against) back to the fit-check inputs for a short grace window
  (`RECENTLY_FREED_MEMORY_GRACE_SECONDS`), so the placement admits immediately;
  the credit expires so a genuine shortfall reasserts, and the worker's live
  pre-load fit guard (#383) remains the OOM backstop.

- **A logprobs request to a llama.cpp node without `logits_all` now fails with a
  clear error instead of silently returning none (#385).** Per-token logprobs on
  the llama.cpp engine require loading the model with `logits_all=True`, which is
  off by default because it pre-allocates a large logits buffer; a request asking
  for `logprobs`/`top_logprobs` against a node that did not enable it used to
  "succeed" with empty logprobs. The runner now refuses such a request with an
  actionable message naming `SKULK_LLAMA_CPP_LOGITS_ALL=1` (delivered to the
  client as an error chunk), so the limitation is legible rather than silent.
  Enabling logprobs remains opt-in per node (`SKULK_LLAMA_CPP_LOGITS_ALL=1`,
  buffer bounded by `SKULK_LLAMA_CPP_LOGITS_ALL_N_CTX`).

- **The model store no longer registers a vision GGUF without its projector.**
  A vision GGUF (LLaVA/Qwen-VL/Gemma-VLM style) ships its multimodal projector
  as a separate `mmproj` file; without it the llama.cpp runner cannot load the
  model. The selective store download already keeps the projector glob, but a
  store entry registered before that logic landed could list only the LM quant,
  so staging it to a worker produced an unloadable model that failed only as a
  runner crash at load time on a remote node. The store host now verifies the
  projector actually landed before registering a vision GGUF and refuses with a
  clear error otherwise, so an incomplete vision model can never be registered.
  An existing stale entry self-heals: a download request for an in-store vision
  GGUF whose entry lacks the projector re-downloads (reusing the already-present
  weights and fetching only the missing projector) instead of being
  short-circuited as "complete", so the failure becomes a loud, fixable download
  error on the store host instead of a confusing crash elsewhere.

- **A borderline multi-node placement is no longer refused on sub-GB memory
  jitter (#383).** The master admits a placement on each node's *gossiped* usable
  memory, but the worker's pre-load guard re-measures *live* at load; on a tight
  multi-node split the live reading can sit a few hundred MB below the admitted
  estimate, so the worker refused a placement the master had just admitted, the
  master could not re-place wider (no spare node), and the model never loaded. A
  24B model split across a 3-node ring was observed refusing at the load re-check
  by 0.2GB (2%). The footprint already includes a 1.30x overhead factor, a full
  KV reservation, and a flat floor, so a marginal miss is within that pad; the
  guard now applies a 10% fit tolerance and refuses only on a shortfall beyond
  it (the signature of a node that genuinely lost memory since admission),
  preserving the leak-on-OOM guard while letting borderline placements load.

### Added

- **The cluster topology now shows per-node health with the fix (#388).** When
  the master recovers a node that could not pull its shard or whose download
  failed, the reason used to be invisible: the node looked normal while
  placements quietly routed around it. `GET /state` now carries a derived
  `nodeHealth` map (per node: a `level` of `ok`/`warn`/`error` plus `reasons`,
  each with a `message` and a `remediation`), computed read-only from state
  already in the response (terminal download failures, low or full models-volume
  disk, and late heartbeats) so it adds no new polling or gossip. The dashboard
  renders an amber/red badge on the affected node whose hover names the problem
  and how to fix it.
- **macOS Local Network permission is now diagnosable and the denial warning is
  actionable (#267).** macOS attributes Local Network access to the responsible
  app in the launch chain (a terminal when run interactively, "Python" over SSH
  / launchd / headless), so generic "grant the app you launched from" advice is
  the part users get wrong. A new read-only probe, `uv run
  skulk-macos-local-network-probe` (text or `--json`), walks the process tree,
  resolves each process's nearest `.app` bundle identity, runs the existing
  reachability check, and reports exactly which identity macOS will attribute the
  grant to. A companion `skulk-build-macos-local-network-probe-app` builds a
  throwaway ad-hoc-signed probe `.app` (with `NSLocalNetworkUsageDescription`) to
  compare terminal vs Skulk-named attribution during development. The startup
  DENIED warning now names the detected app to enable (e.g. "enable 'iTerm2'" or
  "enable 'Python'") instead of generic advice, and points at the probe command.

- **Auto-imported Qwen3 reasoning models no longer return empty content (#384).**
  A Qwen3-family model with no built-in card (a fresh quant imported on demand,
  e.g. `Qwen3.6-35B-A3B-nvfp4`) arrived with empty capabilities, so its resolved
  profile reported no thinking and no thinking toggle. Thinking is on by default
  for Qwen3, so the model reasoned unconditionally and a normal chat request
  spent its whole token budget on the reasoning channel, returning empty
  `content`. Capability resolution now recognizes the Qwen3 / Qwen3.5 / Qwen3.6
  family (token-delimited `<think>` toggle), so an auto-imported variant is
  treated as toggle-capable and the dashboard/API off-by-default path can
  suppress thinking. Built-in cards keep their explicit declarations, an explicit
  `reasoning` section still overrides the family default, and Coder variants
  (instruct-only, no thinking) are excluded.

- **Context-length and other runner errors now surface as structured errors on
  the Claude, Responses, and Ollama wire formats (#276).** Those adapters
  previously raised on a runner error (a 500 on the non-streaming path) or broke
  out of the stream and then emitted an empty successful completion (a bogus
  success for what is actually a clean request rejection). Since the API streams
  every response (the HTTP status is committed to 200 before generation), each
  adapter now emits a structured error envelope in the body and stops, reusing
  the same `error_chunk_response` mapping the OpenAI chat-completions surface
  uses: a `context_length_exceeded` rejection becomes an `invalid_request_error`
  (400), everything else an internal error (500). No more empty-success-on-error
  for clients feeling out context limits.

- **A runner that never reports after spawn no longer stalls an instance forever
  (#272).** A runner frozen between spawn and its first status report (a
  SIGSTOP, a hang in early import or device init) left the instance stuck in
  pre-init coordination indefinitely: `ConnectToGroup` is only planned once
  every rank has reported, and the crash breaker never tripped because the
  process was alive. The worker now applies a first-status-report deadline
  (`_RUNNER_FIRST_REPORT_DEADLINE_SECONDS`, 120s); a runner silent past it gives
  the instance up through the same circuit breaker, so the placement fails and
  recovers instead of hanging. The deadline is generous enough for slow imports
  and weight mmaps.

- **A rank's failed download no longer wedges a multi-node instance forever
  (#381).** If one rank's model download failed terminally (disk full, a
  transient Hugging Face or network error), the ring still formed and every rank
  waited for all ranks to become load-ready; the failed rank never would, and
  nothing failed or recovered the instance, so it sat "loading" at
  `RunnerConnected` indefinitely until a manual restart. The master's plan loop
  now detects this from replicated state (a not-yet-ready instance whose any rank
  node carries a terminal `DownloadFailed` for the model), fails any in-flight
  request bound to it with the download error surfaced, tears the instance down,
  and re-places the model at the same width excluding the failed node(s). A
  transient or single-node failure self-heals onto healthy nodes; a cluster-wide
  shortfall fails cleanly with the reason (`PlacementError` is terminal, bounding
  recovery to the available nodes) instead of hanging.

- **Node logs are now bounded and cannot fill the disk (#382).** Two paths grew
  without limit: the durable `~/.skulk/logs/skulk.log` rotated only once at
  startup, so a long-lived node grew it forever; and the service-manager capture
  files (`skulk.stderr.log` / `skulk.stdout.log`) accumulated across restarts,
  reaching tens of GB on the fleet. Now `skulk.log` rotates at 100 MB with the
  last few runs kept as compressed archives; the capture files are truncated on
  each restart (keeping a 5 MB tail of the previous run as `*.log.1`, tunable via
  `SKULK_CAPTURE_KEEP_BYTES`); the console sink drops ANSI color when stderr is
  not a terminal so captured logs are plain and greppable; and the service now
  launches at info verbosity by default instead of `-v` debug (the libp2p
  transport firehose was the bulk of the volume). Set `SKULK_VERBOSITY=-v` to opt
  back into verbose logging while debugging.

- **The centralized store now honors a card's pinned GGUF quant on download
  (#344).** A store-routed download re-derived the quant from the model id alone
  (the default preference), so a custom card pinning a non-default quant (e.g.
  `Q8_0`, `Q3_K_M`) silently got the default instead. The store download request
  now carries the card's `gguf_file`: the worker's store client sends it in the
  `POST /models/{id}/download` body, the store fetches that quant's shard group,
  and a pin absent from the repo falls back to the default. Auto-built cards
  (whose pin matches the default) are unaffected.

- **A llama.cpp request between the KV budget and the model's context ceiling is
  now cleanly rejected instead of failing at the runner (#362).** The llama.cpp
  runner allocates its KV cache up front and caps the loaded context to
  `KV_CONTEXT_BUDGET_TOKENS` (8192; `_serving_n_ctx`), but the API's admission
  ceiling (`instance_context_token_limit`) was the memory/card value, often tens
  of thousands of tokens. A request above the budget was therefore admitted and
  then failed or truncated at generation. The admission ceiling for a
  GGUF/llama.cpp instance is now capped to the same budget, so the API returns a
  clear `context_length_exceeded` up front and admission matches what the runner
  serves. (Enabling logprobs lowers the runner window further; this was the
  originally reported case, now subsumed. A node that overrides
  `SKULK_LLAMA_CPP_LOGITS_ALL_N_CTX` *below* the budget remains a narrow per-node
  residual, since the master cannot see node-local env at placement.)

### Added

- **Vision GGUF VLMs now run on the llama.cpp engine (#128).** A vision GGUF
  (LLaVA / Qwen-VL style, with a separate `mmproj` projector) can be served on a
  llama.cpp node: the runner loads the projector through llama-cpp-python's
  multimodal chat handler (the general `MTMDChatHandler` by default, or a
  family-specific handler selected from the card's vision `model_type`), and an
  image request's content is passed inline so the handler splices the image
  features itself. A GGUF repo is marked vision-capable from its `config.json`
  vision section when present, or, for the many GGUF VLM repos that ship no
  `config.json`, from the mere presence of an `mmproj` projector. Validated live
  on an AMD Strix Halo (Vulkan) node serving `Qwen2-VL-2B-Instruct-GGUF`
  (reads text in images and describes structured scenes). `image_token_id` on
  the vision card config is now optional: it is required only by the MLX vision
  path; the llama.cpp handler inserts image features without it.

### Fixed

- **Vision GGUF models now download their multimodal projector (#346).** The
  selective GGUF allow-list (`gguf_allow_patterns`, used by both the direct-
  HuggingFace download and the centralized store) only matched the selected LM
  quant's shard group, so a LLaVA-style vision GGUF fetched its weights but not
  its separate `mmproj-*.gguf` projector, and llama.cpp could not do image
  inference. The allow-list now always includes a `*mmproj*.gguf` glob: it
  matches nothing on a text-only repo (no cost) and pulls the projector on a
  vision repo. Foundation for vision GGUF VLMs on llama.cpp (#128).

### Changed

- **The placement single-node constraint is now a named engine capability
  (#328 groundwork).** The hard-coded "llama.cpp is single-node only" check in
  the planner became `engine_supports_multi_node` (MLX yes; llama.cpp not until
  its RPC backend is wired into the runner). Behavior is unchanged today: a
  model whose only compatible engine is single-node is still pinned to a
  one-node cycle, and a card that also allows a multi-node engine (MLX) still
  places across nodes. This is the single hinge to flip when multi-node
  llama.cpp (RPC) lands.

- **Placement now records the resolved backend on each shard (#330).** The master
  resolves which backend a node will use (the card's `compatible_backends`
  intersected with that node's advertised backends, ordered by
  `backend_preference`) at placement time and stamps the winning tag onto the
  node's shard as `resolved_backend`. The worker reads it at runner-spawn instead
  of re-probing its own backends, so engine dispatch is deterministic from
  replicated state and cannot disagree with the placement decision; it also lets
  a card resolve to different engines per node on a heterogeneous cycle. The
  worker falls back to its local probe when the field is absent (a node whose
  resources had not yet gossiped at placement). Foundation for pluggable engines
  (#284) and multi-node llama.cpp (#328).

### Removed

- **The `EXO_*` environment-variable deprecation runway is gone (#324).** Legacy
  `EXO_*` env vars from the pre-rename (exo to skulk) deployments are no longer
  honored: the package-import alias shim (`skulk/__init__.py`), every `EXO_*`
  fallback in `constants.py` and across the worker/API/store, and the
  `~/.exo` path fallbacks (home dir, model staging, download staging) were
  removed. Only the `SKULK_*` names and `~/.skulk/` paths are read now. The
  whole fleet must run the same Skulk version, so re-set any `EXO_*` vars to
  their `SKULK_*` names before upgrading. (The libp2p private-network pre-shared
  key still derives from the `exo_discovery_network` seed in `swarm.rs`; changing
  that is a wire-compatibility break handled separately as a coordinated
  fleet-wide upgrade.)

### Changed

- **The libp2p private-network key seed is now `skulk_discovery_network` (#324).**
  `swarm.rs` derived the PNET pre-shared key from the literal
  `exo_discovery_network` (an exo-rename residue). The seed is now
  `skulk_discovery_network`. **This is a wire-compatibility break**: a node built
  with the new seed cannot form a libp2p cluster with a node built on the old
  seed, so it must roll out as a single coordinated whole-fleet rebuild and
  restart (do not roll nodes one at a time). The Zenoh data-plane namespace is
  unaffected (it derives from `NETWORK_VERSION` / `SKULK_LIBP2P_NAMESPACE`, not
  this seed).

- **Zenoh data plane is now soft default-on (#315).** The `DATA` topic (per-token
  generation output) uses the Eclipse Zenoh transport by default when a node is
  configured for it. `SKULK_ZENOH_DATA_PLANE` is now tri-state
  (`_resolve_zenoh_enabled`): `1`/`true`/`yes`/`on` forces Zenoh on (still requires
  an explicit `SKULK_ZENOH_LISTEN`, #308), `0`/`false`/`no`/`off` forces gossipsub,
  any other non-empty value is rejected, and
  **unset** is the soft default (Zenoh when `SKULK_ZENOH_LISTEN` is set, else
  gossipsub). A bare node with no Zenoh config (e.g. a fresh `uv run skulk`) stays
  on gossipsub rather than failing the listen requirement, so the listen endpoint
  is the opt-in signal under the default. Control, telemetry, and election planes
  stay on libp2p. Validated by a full e2e suite over Zenoh (coherence across
  dense/MoE single- and multi-node, churn/soak/refusal, master-failover
  continuity).

### Added

- **AMD / Linux GPU nodes can join a cluster and serve GGUF models through
  llama.cpp (#325, #331).** A non-Mac box (validated on an AMD Ryzen AI Max+ 395
  "Strix Halo", `gfx1151`, via the Vulkan backend) joins as a worker that serves
  GGUF models on its GPU alongside Apple Silicon nodes serving MLX. Backends are
  self-describing `<engine>-<compute>` tags (`mlx-metal`, `llama_cpp-vulkan`,
  ...); a model card declares `compatible_backends` (a hard placement filter) and
  `backend_preference` (a soft, graceful-fallback ranking), so a GGUF model lands
  only on a llama.cpp node and an MLX model only on the Macs, automatically. The
  llama.cpp runner is single-node and streams tokens onto the existing data
  plane. See `website/docs/amd-strix-halo-nodes.md`.

- **The llama.cpp engine matches MLX on logprobs and tool calling (#356).** GGUF
  models served on an AMD node support per-token `logprobs` / `top_logprobs`
  (opt-in via `SKULK_LLAMA_CPP_LOGITS_ALL=1`, which loads the model retaining
  per-token logits and caps the served context so the logits buffer stays
  bounded; off by default) and tool calling (a request's `tools` are forwarded;
  a structured tool call is emitted when the model returns one, else its prose).
  Multi-token prediction / speculative decoding remains MLX-only: GGUF models
  advertise no MTP capability, so an AMD node serves plain autoregressive without
  promising a speedup it cannot deliver.

- **Collector-agnostic accelerator telemetry (#353, #354).** Node telemetry now
  carries a vendor-neutral `accelerator` block (vendor / utilization / VRAM /
  power / temperature / clock) filled at the collector boundary: mactop on Apple,
  and a passive-sysfs collector for AMD/Linux GPUs, so a non-Mac GPU node is not a
  telemetry blind spot. The dashboard renders it in a vendor-aware accelerator
  panel.

- **Heterogeneous-node identity in the topology (#355).** A Linux node reports a
  real model / chip / OS (DMI + `/proc/cpuinfo` + `os-release`) instead of
  "Unknown", and the dashboard labels non-Mac nodes correctly rather than
  prefixing "macOS".

- **The model store downloads only the selected GGUF quant (#339).** When the
  store host downloads a multi-quant GGUF repo from HuggingFace on a worker's
  behalf, it now fetches exactly what the direct-HuggingFace path fetches: the
  preferred quant's shard group plus `config.json`, and nothing else (not the
  other quantizations, not `original/*` full-precision weights, not `metal/*`
  artifacts). This matches the selective allow-patterns
  (`resolve_allow_patterns`) the direct path already applies, so a store-routed
  download is no larger than a direct one. Non-GGUF repos are unaffected.

- **GGUF cards can be built from the binary header when no `config.json` is
  present (#327).** A GGUF repo that ships only the `.gguf` weights (no
  `config.json`) now has its structural fields (layer count, hidden size,
  KV-head count, context length) read directly from the selected file's GGUF
  metadata header via a ranged read of the file start, instead of failing the
  card build. Repos that ship `config.json` (most community GGUF repos) still
  use it; the header read is the fallback. Completes the selective-quant GGUF
  download/load path so more llama.cpp repos work without a hand-written card.

- **Zenoh data-plane hardening toward default-on (#308 + #309).** Security
  (#308): the Zenoh session now sets a **namespace** (a collision-resistant
  SHA-256 hash of the exact token libp2p isolates on: `SKULK_LIBP2P_NAMESPACE`
  when set, else the `NETWORK_VERSION` default `v0.0.1`, mirroring `swarm.rs`) so
  foreign peers on a different namespace cannot subscribe to this fleet's `data`,
  restoring parity with the libp2p private namespace; and `SKULK_ZENOH_LISTEN` is
  now **required explicitly** when the plane is enabled rather than silently
  defaulting to `0.0.0.0` (an explicit `0.0.0.0` still works but warns). TLS/ACL
  stay operator-configurable for untrusted networks (documented; not built in).
  Robustness (#309): the DATA plane egresses on its **own outbound loop**, so its
  `CongestionControl::Block` backpressure can no longer stall the shared
  control-plane publish loop (commands/events); and `ZenohSession` publish/
  subscribe no longer hold the publishers/subscribers mutex across the
  `declare`/`put` await, so per-command concurrent publishes don't serialize.

- **Data-plane reorder buffer is now transport-conditional (#279 Phase 3).** The
  per-command `sequence` reorder buffer (the #301 fix for gossipsub reordering
  multi-node output) is now skipped when the DATA plane rides Zenoh, which
  delivers each command's chunks per-publisher FIFO, so output dispatches in
  arrival order, eliminating the per-token buffering/reordering hop. The buffer
  stays ON for the gossipsub default (which reorders). The API selects this from
  the transport (`data_plane_zenoh`, from `SKULK_ZENOH_DATA_PLANE`);
  `SKULK_DATA_REORDER_BUFFER` (`1`/`0`) overrides explicitly. Validated 20/20 on
  a 3-node sampled-MTP coherence matrix with the buffer off. The full removal of
  the `sequence` field and reorder machinery is deferred until Zenoh is the
  default DATA transport.

- **Optional Eclipse Zenoh transport for the data plane (experimental, default
  off).** When `SKULK_ZENOH_DATA_PLANE` is set, the `DATA` topic (per-token
  generation output) rides a Zenoh `peer` session instead of gossipsub; control,
  telemetry, and election planes stay on libp2p. Endpoints are per-node and
  explicit via `SKULK_ZENOH_LISTEN` / `SKULK_ZENOH_CONNECT` (multicast scouting
  off, gossip on, the macOS Local Network Privacy-safe posture). The swap is
  transparent above the transport: `DataChunk`, the per-command `sequence`, and
  the reorder buffer are unchanged, so the two transports are interchangeable
  behind the flag. Publishers use `Reliable` + `Block` on a single priority for
  per-key FIFO. With the flag unset, behavior is identical to before. Foundation
  for #279's data-plane evolution (later phase: removing the app-layer reorder
  buffer once Zenoh's per-publisher ordering is relied on).
- **Zenoh data plane is key-addressed per owner (#279 Phase 2), killing the
  cluster-wide fan-out.** The owning API node stamps its node id on the serving
  command (`owner_node`); the master carries it onto the worker task, and the
  rank-0 supervisor stamps it onto each `DataChunk`. On Zenoh the `DATA` topic
  now publishes to the key `data/<owner_node>` and each node subscribes only to
  `data/<own_node_id>`, so generation output reaches just the owning API node
  instead of every node in the cluster. On gossipsub (flag off) `owner_node` is
  ignored and the topic broadcasts as before, so the transports stay
  interchangeable behind the flag.

### Fixed

- **Placement now counts a unified-memory GPU node's GTT-mapped system RAM, not
  just its BIOS VRAM carve-out, and uses a lighter overhead factor for GGUF.** On
  an AMD APU (Strix Halo / Ryzen AI Max) the GPU addresses the BIOS VRAM
  carve-out plus system RAM through GTT, so a model larger than the carve-out
  runs there. `usable_vram_by_node` now detects a unified-memory node (its GTT
  aperture spans the whole system: `gtt_total_bytes > vram_total_bytes` AND
  `gtt_total_bytes ≥ ram_total`, which a discrete card whose GTT default merely
  equals VRAM does not satisfy) and counts working-set-capped VRAM plus
  GTT-mappable system RAM (minus a 16 GB OS headroom) toward the usable pool. The
  weight-overhead factor is now engine-aware: GGUF/llama.cpp models use
  1.10 (lighter C++ runtime) instead of MLX's 1.30. Together these let large GGUF
  MoEs place on a 128 GB Strix Halo node (e.g. a 58.5 GiB gpt-oss-120B on a node
  with a 64 GiB VRAM carve-out). The worker's local pre-spawn guard mirrors the
  same unified-memory math so it never refuses a placement the master admitted.

- **Placement now admits GPU-offload nodes against their discrete VRAM, not
  system RAM.** The memory fit check capped every node at
  `GPU_WORKING_SET_FRACTION` (0.75) of *system* RAM, a Metal/Apple-unified-memory
  assumption. On a discrete-VRAM node (a Strix Halo box whose BIOS carves 128 GB
  into ~64 GB system + 64 GB GPU VRAM) that refused models that fit fine in the
  64 GB VRAM the llama.cpp/Vulkan engine actually allocates from (e.g.
  `Llama-3.3-70B` at ~40 GB: "needs 54.2 GB but can use 46.1 GB"). Placement now
  detects discrete VRAM from the node's accelerator telemetry
  (`usable_vram_by_node`: AMD/NVIDIA `vram_total_bytes`) and admits against
  `min(vram_total − vram_used, GPU_VRAM_WORKING_SET_FRACTION (0.90) × vram_total)`.
  Apple unified-memory nodes are unchanged (they report no discrete VRAM). This
  is engine-agnostic, so it carries forward to vLLM/CUDA nodes.

- **A large-context GGUF no longer OOM-kills the node on load.** The llama.cpp
  runner loaded models with `n_ctx=0`, which sizes the KV cache for the model's
  full trained context (e.g. gemma-4's 128k) instead of the per-instance context
  budget placement actually reserved memory for. On a memory-tight node (observed
  loading gemma-4-31B on a Strix Halo Vulkan node) the kernel OOM-killed the whole
  worker process, so the instance vanished instead of failing cleanly. The runner
  now bounds `n_ctx` to the KV budget placement actually reserved memory for
  (`KV_CONTEXT_BUDGET_TOKENS`, 8192 tokens), clamped down by the instance's
  admission ceiling (#145) on a smaller node, so the up-front KV cache never
  exceeds what the cluster sized for the placement. (Serving llama.cpp beyond that
  budget needs placement to reserve the larger KV footprint, tracked separately
  with VRAM-aware admission.)

- **llama.cpp logprobs no longer OOM a node on load.** Defaulting the runner to
  `logits_all=True` for logprobs parity made llama.cpp pre-allocate an
  `n_ctx * vocab * 4` logits buffer at the model's full trained context, e.g.
  `131072 * 152064 * 4` = 74 GiB for a Qwen2.5-7B GGUF, failing the load with an
  allocation error. logprobs is now opt-in (`SKULK_LLAMA_CPP_LOGITS_ALL=1`) and,
  when enabled, caps the served context (`SKULK_LLAMA_CPP_LOGITS_ALL_N_CTX`,
  default 8192) so the buffer stays bounded; with it off the served context is
  the instance's admission ceiling, not the model's full trained context.

- **The source-built GPU llama.cpp wheel survives `uv sync` (#358).** On a node
  that declares a GPU llama.cpp backend, the service entrypoint now runs
  `uv sync --inexact`, so a routine sync no longer prunes the out-of-resolution
  source-built wheel (which previously dropped the node to CPU-only until a manual
  rebuild). As a safety net, the node cross-checks a declared GPU backend against
  the actual build (`llama_cpp.llama_supports_gpu_offload()`): if the wheel has no
  GPU offload compiled in, it advertises only `llama_cpp-cpu` so GPU work is never
  routed to a degraded build. `SKULK_AUTO_UPDATE=0` is no longer required as a
  workaround on GPU nodes.

- **The llama.cpp tool path honors cancellation (#357).** A tool-enabled request
  runs one blocking, uninterruptible `create_chat_completion`; it now checks
  cancellation at the boundaries around that call (skip if already cancelled,
  suppress the result if a cancel landed while it ran), so a cancelled tool
  request neither delivers output nor is marked complete, matching the streaming
  path's cancellation semantics.

- **Headless/non-Mac nodes boot without the built dashboard (#333).** A worker
  node with no `dashboard-react/dist` (for example a Linux node with no node/npm
  to build the UI) previously failed to start: `constants.py` resolved
  `DASHBOARD_DIR` at import and raised `FileNotFoundError`, and the API's
  `StaticFiles` mount raised when the directory was absent. `DASHBOARD_DIR` is
  now `None` when the assets are absent and no `SKULK_DASHBOARD_DIR` override is
  set, and the API skips serving the dashboard (logging a notice) while serving
  the full API. Nodes that have the assets, or set `SKULK_DASHBOARD_DIR`, serve
  the UI unchanged.

- **Embedding tasks reach a clean terminal state (#326).** The embedding runner
  held `RunnerReady` across the `TextEmbedding` forward pass, but the supervisor
  asserts the runner is in an active state when it forwards a task's terminal
  status, so a completing embedding task tripped the assertion and aborted the
  event forwarder. The runner now holds `RunnerRunning` across the forward pass
  and returns to `RunnerReady` only after the terminal status is emitted,
  matching the MLX and llama.cpp text runners.

- **The failover-seed event round-trips through the disk event log again.** A
  `StateSnapshotHydrated` (the failover seed, indexed as event 0) is read back
  through the `Event` TypeAdapter, whose `TaggedModel` wrap validator unwraps the
  `{ClassName: inner}` envelope by re-validating the inner payload as a *python*
  object. Under `State`'s `strict=True` that path skips JSON-mode coercion, so
  the ISO datetime strings JSON produced for `last_seen` were rejected
  (`datetime_type`) and `DiskEventLog.read_range` halted at the seed. The
  phantom-node fix (#291) had started re-stamping `last_seen` on the seed, so
  every carried seed now hit this. `State` now coerces `last_seen` strings back to
  `datetime` in a field-scoped `before` validator (it does not force the whole
  model into python-mode validation, unlike a model-level validator). This
  unblocks event-log replay across a failover and is a prerequisite for #279
  Phase 3 snapshot/truncate (snapshots persist a full `State`, `last_seen`
  included).

- **Multi-node generation output is no longer silently reordered (#279 Phase 2b
  sequencing).** #279 Phase 2a moved per-token output (`ChunkGenerated`) off the
  master-indexed control plane (where the monotonic event `idx` gave every chunk
  a total order) onto the best-effort `DATA` gossip topic, which has no ordering
  key. When the producing rank-0 worker and the owning API node are different
  nodes, the gossip mesh can deliver a command's chunks out of order, and the API
  consumed them in arrival order, silently transposing tokens/sub-words in the
  response (`"Question"` -> `"Qesution"`). It was specific to multi-node *sampled*
  speculative decoding (single-node is local/in-order; greedy emits steadily) and
  hit ~90% of responses at temperature 0.2; the model battery never caught it
  because it only checked `finish_reason` and token count, never output
  coherence. `DataChunk` now carries a per-command monotonic `sequence` stamped
  by the producing supervisor, and the API reorders by it in a small per-command
  buffer before dispatch (releasing strictly in order, dropping duplicates). A
  genuinely dropped sequence on the best-effort topic is bounded two ways so it
  can never stall a stream: a size cap skips the gap if chunks pile up behind it,
  and a periodic sweep releases a gap left unfilled for `_REORDER_GAP_FLUSH_SECONDS`
  even when no later chunk arrives to trigger the cap (the dropped-seq-0 case,
  where the stream's own idle backstop never arms because nothing was yielded
  yet). The buffer is created only while a command has a live stream and cleared
  with it, so late chunks after finalize don't leak; the producer drops its
  per-command sequence counter on the terminal chunk for the same reason.

- **Deleting an instance no longer leaks its runner records (unbounded
  `State.runners` growth).** Runner status records were only removed by a
  terminal `RunnerStatusUpdated(RunnerShutdown)`, but that final status is
  unreliably delivered: the worker's Shutdown handler cancels the supervisor's
  event forwarder (`runner.shutdown()`) as soon as the Shutdown task
  completes/times out, usually before the runner process's `RunnerShutdown` is
  forwarded, and on a master-failover teardown the forwarder is torn down
  outright. Every instance delete therefore leaked one `RunnerShuttingDown`
  record per rank (one per node for a multi-node instance), so `State.runners`
  grew without bound over the cluster's lifetime, bloating state-sync snapshots.
  Two changes close it: `apply_instance_deleted` now prunes the deleted
  instance's runner records directly (mirroring `apply_node_timed_out`), and
  `apply_runner_status_updated` ignores updates for a runner that belongs to no
  instance, so the late `RunnerShuttingDown` that races behind `InstanceDeleted`
  can no longer resurrect the record. Deletion is now atomic and independent of
  the shutdown handshake. The actual runner-process teardown is driven
  separately by the Shutdown task, so dropping the status record early is safe.

- **Master failover no longer silently kills a healthy serving instance on a
  memory-tight node.** On a master-election transition the winning node tears
  its worker down (`worker.shutdown()`) and rebuilds it; that cancels each
  `RunnerSupervisor.run()`, whose teardown `finally` reaps the runner process so
  Metal reclaims its wired GPU memory on exit. The teardown was not shielded
  from cancellation, so the first `await` in it (the process join) re-raised
  immediately and the runner process was never reaped, so it lingered holding
  its GPU memory. The replacement worker then planned `CreateRunner` for the same
  carried shard, the pre-load memory guard saw the not-yet-reclaimed memory,
  falsely refused, and the #290 re-place-wider path deleted the carried instance
  (every subsequent request 404'd until a manual re-place). The teardown is now
  wrapped in a shielded `CancelScope`, so the runner process is fully joined
  (memory reclaimed) before `worker.shutdown()` returns and the replacement
  worker admits against true post-reclaim availability. Only bites when the
  election winner also hosts a rank of a carried instance and is memory-tight
  (common on small clusters); restores the documented "survives master failover"
  guarantee. The terminate/kill joins are now also off-thread (`to_thread`)
  instead of blocking the event loop.

- **Data-plane streams can't hang on a dropped final chunk (#279 Phase 2b).**
  Output chunks ride the best-effort `DATA` topic (no replay), so a dropped
  final chunk would leave a streaming response blocked on `receive()` forever.
  `_token_chunk_stream` now applies a per-receive idle timeout
  (`_STREAM_IDLE_TIMEOUT_SECONDS`, 120s): once the first real output token has
  arrived, a gap longer than the timeout closes the stream with a terminal error
  instead of hanging. The timeout wraps only the receive (not the yield), so it
  measures producer silence, never a slow client. Time-to-first-token is left
  unbounded, so a request queued behind a long decode or in a slow prefill never
  trips it (prefill-progress chunks are not treated as output and do not arm the
  timer). A stall whose task has already reached a terminal status is a dropped
  *final* chunk, so it cleans up via the normal `TaskFinished` path; a stall on a
  still-active task sends `TaskCancelled` to tear the stuck runner down (avoiding
  both an orphaned runner and a leaked master task/command mapping).

### Changed

- **Plane separation #279 Phase 2a: generation output chunks move to a data
  plane, off the master.** Per-token output (`ChunkGenerated`) used to flow
  worker → master (index + disk write + cluster-wide rebroadcast) → owning API,
  for data that never mutates `State` and is only ever read by that one API
  node. It now travels a new `DATA` topic as `DataChunk` (`{command_id, chunk}`)
  directly from the serving rank-0 worker to the owning API node, which demuxes
  by `command_id` into the per-command stream queues. The master no longer
  indexes, persists, or rebroadcasts output chunks; the API event log no longer
  records the per-token firehose (it had grown ~54MB in 9 idle hours). This
  removes the per-token master hop + disk write that dominated event-log volume
  and was the #278 storm vector. Inbound vision chunks (`InputChunkReceived`)
  stay on the control plane for now. Producer split lives in
  `RunnerSupervisor._emit`; the API consumes in `API._apply_data`.

- **Plane separation #279 slice 3: observational node readings move to the
  telemetry plane.** `node_identities`, `node_disk`, and `node_rdma_ctl` now ride
  the last-write-wins `TELEMETRY` topic into the node-owned `TelemetryView`
  instead of being event-sourced into `State` (joining `node_resources` from
  slice 1 and `node_memory`/`node_system` from slice 2). They are no longer
  persisted in the event log or carried in the failover seed; `GET /state`
  merges them back in so the dashboard wire shape is unchanged. The
  **connectivity** readings (`node_network`, `node_thunderbolt`,
  `node_thunderbolt_bridge`, and the derived `thunderbolt_bridge_cycles`)
  deliberately stay on the control plane — they define the topology graph
  (`apply()` builds RDMA edges and TB-bridge cycles from them, and the planner
  reads `node_network` for host selection), so they remain ordered rather than
  unordered telemetry.

### Fixed

- **Tight multi-node placements no longer silently vanish (#290).** The
  master admits placements on the gossiped (telemetry-plane, last-write-wins)
  available memory, while each worker's pre-spawn OOM guard reads a fresh live
  GPU-wireable figure at load time. On a borderline split the live reading can
  sit just below what the master admitted, so the master placed a cycle the
  worker then refused, and the instance was torn down ("instance vanished")
  with no recovery. The worker now sends a new `RefuseInstancePlacement`
  command for the memory-refusal case (distinct from a crash or GPU wedge,
  which still `DeleteInstance`), and the master re-places the same model one
  node wider (`min_nodes` = refused width + 1) so each node holds a smaller
  share. The loop is bounded: once even a full-width split raises
  `PlacementError` the master stops at the deletion. Refusals for
  already-removed instances are no-ops, so redelivery and operator deletes are
  safe.

## [1.2.0] - 2026-06-11

### Fixed

- **Abandoned requests can no longer storm the event log into election
  churn (#278).** An idle SequentialGenerator re-reported every
  ever-cancelled task id on every step without pruning the set, and the
  runner supervisor converted each re-report into a fresh
  `TaskStatusUpdated(Cancelled)` + `TaskDeleted` pair — observed live at
  ~800 events/s with 12,000+ events minted for a single dead task. The
  flood drowned replica apply loops, starved liveness into cascading
  elections, and silently lost placements. Five-layer fix: the idle
  generator now reports each cancellation exactly once (preserving the
  forward-looking CANCEL_ALL marker); the supervisor forwards a terminal
  status at most once per task; the master refuses to index task-lifecycle
  events for tasks absent from state (capping any future emitter at zero
  amplification); the event router's delivery retry gains exponential
  backoff and a max-attempts cap instead of unbounded fixed-interval
  resend; and the disk event log refreshes its diagnostic metadata file on
  a coarse cadence instead of one open/truncate/write/close per appended
  event (previously the dominant physical-write term of every indexed
  event, cluster-wide).

- **Long-context requests are rejected cleanly instead of OOM-crashing the
  runner (#145, phase 1).** The within-request KV cache grew one entry per
  token with no bound and no preflight check, so a request whose prompt plus
  output exceeded what the hosting node(s) could hold killed the runner
  mid-generation with an unhandled Metal OOM (SIGABRT, broken stream or 500
  for the client, wired GPU memory leaked). Each placed instance now carries
  a static context-token ceiling — the smaller of the card's advertised
  context length and the KV tokens that fit beside the weight share on every
  hosting node — computed deterministically from gossiped node memory so all
  ranks of a multi-node instance enforce the identical limit. Requests are
  admitted against it before prefill: explicit `max_tokens` overflow and
  window-filling prompts get an OpenAI-style `context_length_exceeded`
  invalid-request error (400 at the API when detectable pre-dispatch), and an
  omitted `max_tokens` is clamped to the remaining window so generation ends
  with `finish_reason: "length"`. Unquantized KV only; quantized-KV budget
  math is phase 2.

- **Instance placements survive master failover (#273).** A newly-elected
  master previously always started its session from an empty state: the
  empty snapshot propagated to every follower, each worker's plan loop saw
  no instances and shut down its healthy runners, and every placed model
  silently became a 404 until an operator re-placed it — a full serving
  outage from a single master restart (found live when a churn test
  happened to bounce the master). The promoted node now seeds the new
  session from its prior replicated state: instances, downloads, node info,
  and the tracing flag carry over, while in-flight tasks, runner statuses,
  topology, and liveness timestamps are deliberately dropped (they are
  session-scoped or must come from live gossip — a carried topology would
  keep a dead node's edges forever). Workers re-create runners for the
  carried instances through the ordinary plan loop, so serving resumes
  after a model-reload-sized gap with no operator action. The master's
  liveness-based instance pruning is suppressed for a 60-second
  topology-settle grace after promotion so carried instances aren't deleted
  while connection gossip is still rebuilding the topology; instances whose
  ranks lived on the dead master are pruned normally after the grace. A
  freshly-booted election winner seeds empty, exactly as before.

- **A stalled distributed group can no longer hang an instance forever, and
  ring transport selection follows operator intent (#265).** Two changes:
  (1) `mx.distributed.init` now runs under a hard deadline (default 120s,
  `SKULK_GROUP_CONNECT_DEADLINE_SECONDS`) — the ring backend with
  `strict=True` blocks indefinitely when a neighbor socket fails its
  post-TCP rank handshake, which left a 4-node placement looping request
  timeouts and cancels for 30+ minutes with no recovery; expiry now exits
  the runner via the wedge path, the worker gives the instance up on the
  first failure, and the fresh placement mints a new ring port (also
  clearing stale-socket handshake collisions from same-port retries).
  (2) VPN/overlay addresses (Tailscale CGNAT `100.64/10` and
  `fd7a:115c:a1e0::/48`, detected by address since utun interfaces gossip
  as "unknown") now rank strictly last in ring transport selection —
  Tailscale exists for external reachability and may be DERP-relayed (a
  ring link between two machines on the same switch was observed riding
  the Dallas relay); a pair with any Thunderbolt/LAN candidate never
  selects the overlay, while genuinely cross-network pairs still work.
  First test coverage for `get_mlx_ring_hosts_by_node` and the transport
  ranking.
- **The Thunderbolt interface label survives classification (#222).** The
  hardware-port parser set "thunderbolt" from the port header, then the
  device-line branch unconditionally rewrote every en2+ device to
  `maybe_ethernet` — and Mac Thunderbolt ports are always en2+, so the
  thunderbolt label could never exist on macOS and the ring's TB-first
  transport priority was dead code (it worked only because maybe_ethernet
  happened to outrank ethernet). The downgrade now applies only to the
  genuinely ambiguous case (a generic "Ethernet Adapter" port on en2+, which
  may be a USB dongle); specifically-classified ports keep their labels, and
  unclassified ports (e.g. an iPhone tether) stay at lowest priority instead
  of being promoted.
- **Peer churn can no longer crash healthy bystander nodes (#266).** When a
  master transition replaced the worker, the telemetry forwarder exited first
  (its event stream closes), and the InfoGatherer's next send raced into the
  closed channel — the unhandled `BrokenResourceError` took the entire
  process down (observed twice in one night, once on the cluster hub). A
  closed/broken telemetry channel is now treated as the stop signal it is:
  the gatherer exits cleanly (the replacement worker brings a fresh one), and
  the per-monitor `except Exception` blocks — which exist to survive flaky
  *gathering* — explicitly re-raise channel closure instead of swallowing it
  and spinning on a dead channel.

- **GPU-wedge runner deaths are no longer retried (wired-memory leak).**
  Contrary to every other crash class, a runner hard-exited by the warmup
  deadline watchdog while its main thread is parked in a faulted Metal eval
  does NOT get its wired GPU memory reclaimed on exit — measured live
  (4-node matrix testing, 2026-06-09): each wedge-exit left ~5GB wired
  behind, recoverable only by reboot, and two automatic retries cost a 24GB
  node ~10GB. Worse, wedges take ~300s each so the 3-failures-in-60s crash
  breaker never trips — unattended, the relaunch loop leaks the node to
  death. The watchdog now exits with a distinct code (`WEDGE_EXIT_CODE`),
  the supervisor marks the failure (`gpu-wedge-deadline` in the runner's
  failure message — a string marker keeps the gossiped status type
  wire-compatible during rolling upgrades), and the worker gives the
  instance up on the FIRST wedge death with a log that names the leak and
  the reboot remedy.
- **`MLX_METAL_FAST_SYNCH` now defaults OFF cluster-wide.** The old ON default
  had no measured upside (vanilla dense decode: 20.8 tok/s off vs 20.7 on) and
  a catastrophic failure mode for any model without a curated card pin:
  hybrid-SSM models wedge at warmup under the flag — gpt-oss hit the 300s
  warmup deadline (#236, card-pinned off on 2026-06-07) and NemotronH-9B did
  exactly the same (#259) — and the resulting deadline kill mid-GPU-work leaks
  ~5GB of wired memory per attempt and degrades the node until reboot. With
  the flag off, Nemotron-Nano-9B warms in seconds and decodes at 19+ tok/s.
  All NemotronH/Nemotron-3-hybrid and gpt-oss cards also carry an explicit
  `runtime.metal_fast_synch = false` pin now, and any model that measurably
  benefits from FAST_SYNCH can pin it on per card; the operator override
  (`--fast-synch`/`--no-fast-synch`) is unchanged.

### Added

- **The dashboard now shows speculative-decoding status per instance.** Active
  instances with an MTP sidecar or assistant drafter display an `MTP D{n}`
  badge next to the status badge — depth from the card's `mtp_max_depth`,
  with the drafter kind spelled out in a hover tooltip.
  The status is derived from the model card's runtime section already present
  in cluster state — the rank-invariant source of truth for whether drafting
  engages (#254) — so no new wire data is needed. Cards that block multi-node
  speculation (`speculative_multi_node=false`) show no badge on multi-node
  placements, matching the runtime behavior.

### Fixed

- **Tensor-parallel placements of sidecar-MTP models no longer crash on the
  first request (#263).** The decider-only sidecar load introduced with the
  explicit lockstep protocol (#254) regressed tensor placements: draft
  logits go through the TP-sharded lm_head, an all-rank collective the idle
  receiver ranks never join, so the lone TP decider GPU-timed-out inside its
  first draft round and SIGABRT'd in the Metal completion block while the
  receivers hung to the eval watchdog. Deterministic on a homogeneous M4
  pair (plain TP decode on the same instance worked; only the speculative
  path wedged). Tensor placements now load the sidecar on every rank and
  draft rank-symmetrically — the same envelope assistants use on TP and the
  configuration the published +31% TP benchmark measured — while the
  decider protocol remains in force for pipeline placements. The drafter
  agreement still disables speculation symmetrically if any TP rank fails
  to produce a working drafter.
- **Placement no longer refuses models whose weights sit in the macOS file
  cache (cache-deflated availability).** The gossiped `ram_available` came
  from mactop's `available` (free + inactive + speculative), which counts
  reclaimable file cache as *used* — so immediately after downloading a model,
  availability was deflated by roughly the model's full size and placement
  refused fits that run comfortably (observed on a 24 GB node: 11.6 GB of
  just-downloaded weights in cache dropped "available" to ~12 GB while
  ~14.6 GB was genuinely wireable). On macOS, `ram_available` is now the
  GPU-wireable figure `total − wired − anonymous − compressor`, taken from a
  `vm_stat` snapshot alongside each telemetry sample; macOS reclaims file
  cache the moment Metal wires pages, so this is what a runner can actually
  use. The metric deliberately does not credit compression of idle anonymous
  memory, preserving the conservative posture of the oversized-placement OOM
  hardening (#243). The worker's local pre-spawn fit guard judges with the
  same metric (it previously used psutil's free + inactive, which would veto
  the very placement the master had just correctly admitted). Value-only
  change to the gossiped figure — the wire shape is unchanged, so
  mixed-version clusters interoperate.
- **Dashboard deep links and browser refresh no longer 404.** The dashboard is
  a SPA that restores its active view from the URL path, but the API served
  `index.html` only at `/` — refreshing on `/chat` (or following a shared link
  to `/cluster`, `/model-store`, `/operator`) returned a bare
  `{"detail":"Not Found"}`. The API now serves the SPA shell for the four
  client routes (kept in sync with the dashboard's `NavRoute`).
- **Multi-node speculative decoding no longer crashes on heterogeneous
  clusters (explicit cross-rank lockstep, #252/#254).** Distributed MTP kept
  ranks in sync by *assuming* every rank independently recomputed bit-identical
  accept/reject decisions from its own logits. Heterogeneous chips break that
  assumption (M5 vs M4 GEMM kernels differ; M5 additionally runs reduced-
  precision B≥2 matmuls), so mixed-chip pipelines desynced: ranks committed
  different token counts, fell out of the collective schedule, and one rank
  SIGABRT'd inside the Metal command-buffer completion block while the other
  waited on a `MTLSharedEvent` forever. The protocol is now explicit: the
  decider (last) rank alone holds the drafter, drafts, and decides; draft
  tokens and the per-round accept outcome (`[prefix_len, bonus_token]`) are
  broadcast via fixed-shape `all_sum` collectives, and receiving ranks apply
  the broadcast decisions to their own cache slices without ever sampling or
  comparing logits. The same applies to the request's first sampled token, and
  the non-MTP pipeline fallback broadcasts each step's token the same way (its
  per-rank sampling silently desynced heterogeneous ranks). Sidecar drafter
  weights now load only on the decider rank (matching assistants), saving
  drafter memory on every other rank.
- **Crash circuit breaker now trips once per crash loop, not once per failure.**
  `CrashWindow.record()` is edge-triggered: it returns `True` only when the
  in-window failure count *crosses* the threshold and stays latched (returning
  `False`) until the window drains below it, and `_give_up_on_instance` no
  longer clears the window. Previously the trip was level-triggered and the
  window was cleared on give-up, so a doomed instance lingering in replicated
  state before its `DeleteInstance` landed could re-accumulate and re-trip,
  emitting duplicate `DeleteInstance` commands and "giving up on instance" logs.
  `InstanceId`s are unique, so the retained failure history can never collide
  with a future instance, and the worker reclaims breaker entries for deleted
  instances each planning tick (`CrashWindow.retain`) so the history can't grow
  unbounded. (Follow-up to #243.)
- **Oversized model placements no longer brick a node.** Placing a model whose
  shard does not fit a node's memory previously passed an over-optimistic
  admission check (1.05x weights against gossiped `ram_available` only),
  OOM-aborted during load, and orphaned wired GPU memory reclaimable only by
  reboot — then the worker relaunched the doomed runner every ~1.5s with no
  backoff, compounding the leak (the GLM-4.7-Flash incident). Three changes
  harden this: (1) placement estimates a realistic footprint — weights x 1.30,
  an explicit KV-cache reservation for an 8192-token planning budget, and a
  per-node cap at the Metal GPU working-set ceiling (~75% of RAM) — and shards
  proportionally to fit heterogeneous clusters rather than refusing what fits;
  (2) the worker refuses a shard that won't fit *local, current* memory before
  spawning the runner, failing cleanly instead of OOM-aborting; (3) a crash
  circuit breaker gives up after 3 runner failures within 60s and deletes the
  instance instead of looping. Estimation lives in one shared module
  (`skulk.shared.models.memory_estimate`) used by both the master admission
  check and the worker guard so the two never disagree.
- **macOS node telemetry no longer crashes MLX inference (macmon → mactop).**
  Skulk's `InfoGatherer` spawned `macmon` at 1 Hz for hardware metrics on every
  macOS node. macmon reads the GPU via IOKit/IOGPUFamily — the same interface
  Metal uses for command-buffer completion — so sampling it concurrently with
  an in-flight MLX command buffer put the GPU into an error state that
  `mlx::core::gpu::check_error` threw inside the Metal completion-dispatch
  block: either an uncaught `abort()` (SIGABRT) or a silent GPU hang. On macOS
  the wedged GPU then starved WindowServer past its watchdog and **rebooted the
  node**. (Confirmed upstream as exo-explore/exo#2088 / #1823.) Replaced macmon
  with [`mactop`](https://github.com/metaspartan/mactop), which reads Apple's
  IOReport/SMC counters (not IOGPUFamily), needs no root, emits newline-
  delimited JSON (`--headless --format json`), and exposes a superset of the
  metrics (GPU util %, power breakdown, temps, DRAM bandwidth, system RAM).
  Validated on M4 hardware running sustained MLX inference with zero crashes.
  Provisioning moved with it: `README`/`CONTRIBUTING` (`brew install mactop`),
  the nix dev shell + package wrapper (`pkgs.mactop`), and the PyInstaller
  bundle. When mactop is absent the gatherer still falls back to psutil for
  memory. (mactop's reported `available` RAM equals `total − used`, the same
  figure macmon derived, so placement margins are unchanged.) The gossiped
  `NodeGatheredInfo` event keeps a decode-only `MacmonMetrics` shim so a
  newly-upgraded node still applies telemetry from macOS workers on the
  pre-mactop build during a rolling upgrade. A blank or unparseable line from
  mactop is now skipped rather than tearing down and respawning the subprocess.
- **Topology GPU bar no longer renders 100× too high.** The dashboard treated
  `SystemPerformanceProfile.gpuUsage` (a 0–100 percent) as a 0–1 fraction and
  re-multiplied it by 100, so e.g. 8.66% GPU showed as 866%. It is now
  converted to a fraction when populating the node's monitoring snapshot.

### Changed

- **The codebase is now Skulk all the way down (exo -> skulk rename).**
  The Python package is `skulk`, the Rust bindings crate is
  `skulk_pyo3_bindings`, the wire identity fields are
  `skulkVersion`/`skulkCommit`, and environment variables use the
  `SKULK_*` prefix. Backward compatibility is explicit: legacy `EXO_*`
  environment variables are aliased at startup (an explicit `SKULK_*`
  value always wins), the legacy `exo.yaml` config name is still
  honored, a populated pre-rename `~/.exo/staging` directory keeps
  being used when staging is unconfigured, and the dashboard migrates
  saved favorites/recents once. The deprecated `uv run exo` alias is
  removed — the command is `uv run skulk`. Upstream attribution (the
  "forked from exo" acknowledgment and exo's license copyright) is
  deliberately preserved.

### Fixed

- **Empty `messages` (and non-positive `max_tokens`) are rejected with
  400 instead of crashing the runner.** An empty message array was
  accepted, then `apply_chat_template([])` raised `IndexError` inside the
  runner — taking down the process serving that instance. A single
  renderability guard at the shared text-generation dispatch chokepoint
  (covering chat, Claude, Ollama, and Responses wire formats) now returns
  400 before the request reaches a runner. Found by the post-rename
  torture battery (#233).

- **Requests no longer hang when a node dies mid-generation.** When an
  instance was lost (node disconnect, crash, or deletion with a request
  in flight), the master tore the instance down but left the task
  orphaned — the API never received a terminal chunk, so the open HTTP
  connection hung until the client's own timeout. The master's plan loop
  now emits `TaskFailed` for in-flight API tasks whose instance is gone,
  and the API turns that into a terminal error chunk: streaming
  responses close with an error event, non-streaming requests return a
  500. Found by the 2026-06-07 node-kill drill (#223).

- **Master failover no longer strands open requests.** Killing the
  master mid-generation starts a new cluster session that cannot carry
  the old session's tasks, and the API's session reset replaced its
  command-queue maps without closing the old streams — a guaranteed
  permanent hang that the orphaned-task sweep above structurally could
  not cover. The API now fails every open command stream at the session
  boundary with an error explaining the session changed and asking the
  client to retry.
  Verified end-to-end: clients receive the error within ~4–6 seconds of
  a node kill (master or worker rank), versus an indefinite hang before.

### Changed

- **Speculative-decoding draft depths are now per-card measured optima.**
  A production depth sweep (3×200-token greedy A/Bs per cell) moved the
  gemma E-series assistant cards from depth 3 to depth 2 — E2B-8bit
  37.7 → 54.0 tok/s (+43%, was +20% at depth 3), E4B-8bit 19.5 → 25.4
  (+30%) — and Qwen3.5-27B from depth 2 to depth 1 (6.3 → 10.5 tok/s on
  a 2-node pipeline, +67%; depth 2's run-to-run spread was the
  GDN/SSM deferred-replay tax). Mechanism: on M4-class GPUs verifying up
  to 2 candidate tokens per step is effectively free, but each candidate
  beyond width 2 costs ~36% of a full forward pass — so drafting deeper
  than depth 2 over-spends on every model measured, and SSM-hybrid
  models pay an additional replay tax even at depth 2. Rule of thumb:
  gemma assistant cards depth 2, Qwen GDN sidecar cards depth 1.

- **A bare `repetition_penalty` no longer crashes the runner.** Requests
  carrying `repetition_penalty` without `repetition_context_size` passed
  the request's None straight into mlx-lm's processor builder, overriding
  its default of 20; the penalty processor's `tokens[-None:]` slice then
  raised and killed the runner on the first penalized request. Both call
  sites now coerce None to the default. Found by the 2026-06-06
  before/after benchmark matrix; applies to every model and every client
  that sends a penalty alone (many do by default).

### Added

- **Staged model copies now have a lifecycle, and nodes can report their
  storage.** With the model store on, staged copies previously survived
  instance deletion and node crashes forever (58-70 GB piles; one node
  died of a full disk in the launch smoke). `cleanup_on_deactivate` now
  defaults to true with a recent-use grace budget
  (`staging_keep_recent_gb`, default 40 GiB): when an instance shuts
  down — and at node startup, which reconciles copies orphaned by a
  crash — not-in-use staged models are kept newest-first by last use up
  to the budget and evicted beyond it. In-use detection includes
  companion repos (MTP sidecar / assistant / vision weights) of active
  models, so eviction can never corrupt a live runner. The grace budget
  is deliberate: node deaths, restarts, and repeated place/delete cycles
  of the same model do not re-pay the staging copy. New
  `GET /store/storage` returns the local node's breakdown: staged models
  with size/last-use/in-use, event-log bytes, and disk free.

### Fixed

- **Event logs can no longer eat the disk or kill nodes on a full one.**
  The API-side event log — which records per-token chunk events and backs
  only the `GET /events` diagnostic — had NO retention and grew for the
  life of the session (54 MB in 9 idle hours on every node; the file a
  node died writing during the launch smoke). It now ring-compacts past
  256 MiB, keeping the most recent 20k events. Archive rotation is capped
  by total bytes (1 GiB) in addition to count — five archives of
  unbounded size defeated the count cap in practice (3.5 GB observed).
  The remaining unguarded ENOSPC sites (`DiskEventLog.__init__` and
  `compact()` — the former is exactly where a node died) now degrade to
  the counting-only mode instead of crashing, and a proactive free-space
  floor (2 GiB, checked every 1024 appends) degrades persistence BEFORE
  the disk hits zero — a master on a full disk previously throttled the
  whole cluster to ~0.5 tok/s before dying. Log noise that bloats piped
  logs was also trimmed (per-minute download-coordinator path dumps), and
  the speculative-decoding enable line now reports the card's actual
  draft depth instead of a hardcoded "(D=1)".
- **Speculative decoding now engages for models that were already on
  disk.** Three of the model-store downloader's four resolution paths
  (already-staged fast path, store staging, direct-from-store) returned
  the base model without fetching the card's companion repos (MTP
  sidecar / assistant model / vision weights) — a staged model would
  load and silently run without speculation (observed in the launch
  smoke). Every `ensure_shard` resolution now also ensures companions
  through the same store-first path (so sidecars are served from the
  store when present), optional-companion fetch failures (MTP
  sidecar / assistant) log loudly without failing the base load, while
  split vision weights stay load-bearing, and the previously triplicated companion
  construction in the HF downloader is shared via
  `companion_download_specs`.
- **A wedged warmup no longer silently disables a node.** A faulted
  Metal eval can park warmup forever at 0% CPU (uninterruptible from
  Python); the runner then sat in `RunnerWarmingUp` indefinitely while
  every API request queued and timed out with no surfaced error. Warmup
  now runs under a hard deadline (default 300s,
  `SKULK_WARMUP_DEADLINE_SECONDS`): on overrun the runner logs a
  CRITICAL diagnosis (including the reboot-if-GPU-wedged guidance) and
  exits, the supervisor reports `RunnerFailed`, and the node keeps
  dispatching.
- **Disabled speculation is no longer near-silent.** Requests carrying
  logits processors (typically a `repetition_penalty` — some client
  libraries send one by default) fall back to plain decode; that
  fallback now logs a WARNING naming the cause and the fix instead of
  an easy-to-miss INFO line. The gemma 4 E-series pipeline rejection
  also explains itself in operator terms (place on a single node)
  instead of internals-speak.

- **Multi-node placement is now reliable and placement failures are
  visible.** Four compounding issues fixed in the placement path:
  (1) memory admission is per node instead of summed across the cycle —
  Tensor sharding splits weights evenly, so a 16+24 GB pair whose *sum*
  covered the model could be admitted with the even split overloading the
  smaller node; (2) admission requires runtime headroom
  (weights x 1.05 + 256 MB per node) on top of raw weight bytes — an
  exact weights-equal-free-memory fit previously produced a silent
  thrash (observed: 12-token prefill in 1230 s) instead of a refusal;
  (3) placing immediately after cluster formation no longer fails with a
  false "insufficient memory" — cycles touching nodes whose memory info
  has not been gossiped yet are now reported as info-pending, and
  `POST /place_instance` waits up to 15 s for the info before returning
  503; (4) impossible placements now fail loudly at the API with the
  specific typed reason (400) instead of returning "Command received"
  and silently failing on the master, leaving clients with unexplained
  404s. The old catch-all "No cycles found with sufficient memory" error
  (which fired for topology gaps, exclusions, startup races, AND real
  shortfalls alike) is split into per-stage `PlacementError` messages
  that include the per-node GB arithmetic.
- **Production MTP no longer runs ~20-46x slower than plain decode.**
  `FAST_SYNCH_CLUSTER_DEFAULT = True` silently applied
  `MLX_METAL_FAST_SYNCH=1` to every MTP runner, collapsing the
  speculative loop (Qwen3.5-9B-4bit on M4, mlx 0.31.2: 27.7 tok/s with
  the flag off vs 0.6 tok/s with it on) while leaving vanilla decode
  untouched (20.8 vs 20.7 tok/s). Probe harnesses never set the flag,
  which is why isolated measurements showed +26-50% while the production
  stack inverted. `resolve_metal_fast_synch` now defaults to OFF for any
  card that declares a speculation mechanism (`mtp_heads`,
  `mtp_sidecar_repo`, or `assistant_model_repo`); operator overrides and
  explicit card pins keep their precedence. Validated end-to-end:
  production Qwen 9B MTP went from 9.5 to 27.7-28.5 tok/s on a 16GB M4
  (~+55% over plain decode, 82-84% acceptance).

### Added

- Distributed gemma4 assistant drafting + gemma4 pipeline sharding (#201
  Track 2b): assistant-model speculation now runs on pipeline placements
  via LAST-RANK drafting — the assistant cross-attends the target's last
  full-attention/sliding KV layers (resident on the final slice by
  construction) and post-norm hidden (already all-gathered), and every
  rank joins one fixed-shape `all_sum` per round carrying the draft
  tokens (plus the drafter's effective distribution under sampling, so
  ratio-acceptance runs identically everywhere; drafting-rank draws use
  explicit per-round keys to keep global RNG streams aligned). Assistants
  load on the last pipeline rank only. En route, gemma4 pipeline sharding
  itself was made to work at all: decoder layers return (hidden, kvs,
  offset) tuples the wrappers now carry, and layer_types/previous_kvs/
  make_cache are re-keyed per slice — slices cutting a KV-sharing edge
  (E-series) fail loud, since those models fit single-node anyway. Two
  cross-attention correctness bugs found and fixed (masked by mlx-vlm's
  native rollback): deferred replay starved assistant drafters of
  committed tokens (74% -> 28% acceptance; the Drafter protocol gains
  `reads_target_cache` and the loop flushes immediately for such
  drafters), and the drafter held a COPY of the cache list that froze its
  view at the first reject-restore (progressive 56% -> 26% decay; it now
  holds the live sequence). Gemma4 coverage grew three validated cards —
  12B (2.03x single-node, 95%-of-single across 2 nodes), 31B (2.48x
  single; the pipeline flagship: 2x16GB nodes lift vanilla 3.8 -> 5.6 and
  MTP reaches 7.75 tok/s), E2B (1.56x) — with assistant-pipeline lockstep
  regression tests (greedy + sampled) alongside the Track 1/2a ones.

- Pipeline speculative decoding (#201 Track 2a): sidecar MTP now runs on
  pipeline-sharded placements with NO new distributed protocol — pipeline
  decode was already rank-symmetric (`pipeline_auto_parallel` slices only
  layers; embed/norm/head load in full everywhere, and decode-mode
  `PipelineLastLayer` all-gathers the final hidden to every rank), so the
  existing bonus-driven loop runs identically on each rank. Lockstep
  validated like Track 1: greedy byte-parity (depths 1-2) and
  seeded-sampled trace parity over 300-token generations, on a localhost
  ring and on real two-node hardware; pipeline acceptance matches
  single-node (79% on the 2B — full-precision slices, unlike TP's
  resharded reductions). Drafting is rank-local against replicated
  embed/head and overlaps the pipeline's sequential bubble; the K+1-wide
  verify pays one hop-set regardless of width, so inter-node latency
  amortizes per committed token — the placement where speculation helps
  most. Safety rails: a per-request `all_sum` keeps the speculate-or-not
  choice symmetric when a rank's sidecar is missing, and mid-request
  drafter failures abort loudly on multi-rank placements instead of
  silently forking the collective schedule. Assistant drafters (gemma4)
  cross-attend the target's KV — which a pipeline shard only holds for
  its own layers — and stay single-node/TP (#201 Track 2b). Pipeline
  lockstep regression tests (greedy + sampled) join the TP ones.

- Tensor-parallel speculative decoding (#201 Track 1): the #200
  single-node guard is lifted for TP placements after lockstep was
  validated on real hardware — greedy byte-parity and seeded-sampled
  trace-hash parity across ranks, on both a localhost ring and a real
  two-node cluster (kite1+kite2, Qwen3.5-2B TP=2, 150–300-token
  generations, multiple seeds, depths 1–2). The two invariants that make
  it safe: TP collectives give every rank identical logits (embeddings
  and lm_head are replicated, only layer internals shard), and
  `mlx_generate` already seeds the RNG per request from the shared task,
  so sampled accept/reject draws are aligned with zero extra
  communication — unseeded ranks fork on the first draw (measured), which
  is why the probe validates the production seeding contract. Pipeline
  placements still disengage speculation pending the distributed
  draft/verify design (#201 Track 2). A two-rank lockstep regression test
  (greedy + sampled) guards the invariant.

- Bonus-driven MTP rounds: the speculative loop was restructured to the
  cadence the reference implementations use — every round verifies
  `[bonus, drafts]` in one forward and the very next round drafts from
  the correction position, instead of skipping post-correction drafts
  (statistically the easiest ones; the old cadence forfeited ~25pp of
  acceptance on identical inputs). Two companion optimizations close the
  hybrid-SSM gap the new cadence exposed: *deferred replay* (on a reject,
  restored-but-committed tokens ride at the front of the next verify
  forward instead of paying a dedicated replay pass — extra verify width
  is free on memory-bound decode, measured 46.6ms 2-wide vs 47.8ms 1-wide
  on Qwen3.5-9B) and *quantize-on-load sidecars* (the builder quantizes
  the bf16 sidecar block + fc to the target's `(group_size, bits)`; the
  unquantized block was ~10.7ms of the round budget). Re-measured
  2026-06-05 on M4/24GB, superseding all earlier figures in this section:
  Qwen3.5-9B 79% acceptance / 1.38x greedy (1.43x at T=0.7), Qwen3.5-27B
  depth-2 82% / 10.5 tok/s (1.87x), gemma-4-26B-A4B 84% / 35.1 tok/s
  (~2.2x vs warm vanilla), gemma-4-E4B depth-3 1.86x — beating upstream
  mlx-vlm (1.43x/1.66x) on identical artifacts. This RETRACTS the earlier
  "chained depth does not pay on quantized targets" finding below: it was
  an artifact of the old cadence's skipped drafts, not of quantized
  hiddens (E4B's carded depth is now 3).

- Gemma 4 assistant speculative decoding (gemma4-mtp Phase C): the
  separate 4-layer assistant models Google publishes per Gemma 4 target
  now draft through Skulk's Drafter protocol — the assistant cross-attends
  over the target's KV cache (shared-KV extraction with RotatingKVCache
  temporal restore), consumes the target's post-norm hidden, and loads
  bf16-enforced when a card declares `assistant_model_repo` (single-node,
  same #200/#201 envelope; forces SequentialGenerator like MTP). Measured
  on M4/24GB: gemma-4-26B-A4B-it-4bit 55% acceptance, 28.8 tok/s vs
  15.5–17.8 vanilla (1.6–1.85×); gemma-4-e4b-it-8bit 48%, 1.26×. Notable
  finding: chained depth does NOT pay on quantized targets (the assistant
  is trained against bf16 hiddens; chain acceptance decays to ~30% and MoE
  verify cost grows with block size) — depth 1 is the default and the
  measured optimum for the carded quants. Also fixed en route: the
  pre-norm trunk wrapper is gated to qwen-shaped trunks (it would build
  wrong masks for gemma4's sliding/full layers), and the companion-repo
  download-completeness gap (#185 flag) — cached bases now fetch newly
  declared sidecars/assistants.
- Sampled-decoding support for MTP speculative decoding (issue #180 item 1):
  at temperature > 0 the loop switches from argmax-prefix acceptance to
  Leviathan-Chen probability-ratio rejection sampling over the *effective*
  sampler distributions (temp + top_p + min_p + top_k, computed by reusing
  mlx-lm's own filter functions so they cannot drift), with residual
  resampling on reject — distribution-preserving by construction and
  verified by a 40k-draw statistical unit test. Depth is forced to 1 under
  sampling (the drafter's internal chain is greedy). Measured at T=0.7
  with default min_p: 9B 87% acceptance / 1.33x, 2B 71% / 1.09x; greedy
  path regression-checked identical. MTP previously disengaged entirely
  for any temperature > 0 — this extends every speedup to default-
  temperature chat traffic.

- Depth-K chained MTP drafting: the speculative loop now verifies up to
  `mtp_max_depth` chained drafts in a single K+1-token forward, committing
  the longest matching prefix (plus the verifier's correction on partial
  rejects). The Qwen drafter chains by recursing its block on its own
  output hidden — measured conditional acceptance decays fast beyond one
  step (86.8% / 39.2% / 28.2% at depths 1-3 on Qwen3.5-9B), so depth 2 is
  the practical ceiling for the single trained block; deeper gains need
  heads trained for chaining (or the Gemma 4 assistant drafter). On a full
  accept the next main token now comes straight from the verify logits,
  eliminating a redundant lm_head pass per accepted cycle. Measured: the
  recompute fix alone lifts 9B depth-1 from 1.20x to 1.30x; depth 2 lifts
  Qwen3.5-27B to 1.92x (from 1.73x, 78% chained acceptance, parity OK) but
  is SLOWER than depth 1 on the 9B (1.15x vs 1.30x) — depth only pays when
  the trunk dwarfs the drafter, so `mtp_max_depth` is set per card (2 on
  27B-class, 1 elsewhere).

- Phase 2 MTP speculative decoding behind a modular `Drafter` protocol
  (`src/skulk/worker/engines/mlx/drafters/`). The generation loop now talks to
  a mechanism-agnostic drafter seam (`begin_request` / `observe` / `draft`)
  so Qwen sidecar heads, DeepSeek heads, and the planned Gemma 4 assistant
  drafter all plug into the same verify/accept/reject machinery. The
  Qwen3.5 drafter applies the three empirically isolated fixes from issue
  #192 — +1.0 zero-centered norm shift, `embed_first` fc concat order, and
  running the sidecar's `mtp.layers.0` transformer block with a private KV
  cache — measured live at ~58–66% draft acceptance on Qwen3.5-2B (0%
  before). Model cards gain optional `mtp_norm_convention` /
  `mtp_concat_order` runtime overrides keyed to layout-detected family
  defaults, and the loop logs a periodic `MTP acceptance so far` line as
  the production acceptance signal. Model cards for Qwen3.5 2B-4bit (69%
  acceptance, 1.26x), 9B-MLX-4bit (88%, 1.20x), and 27B-4bit (1.75x) now
  declare MTP sidecars, validated by a per-model sweep plus a 10-prompt
  exact-attempt acceptance suite. All shipped sidecars use base heads: a
  750-draft/arm comparison measured base vs instruct heads as
  statistically indistinguishable (87.6% vs 87.3% on 9B), so one base
  sidecar serves every variant of a backbone. Qwen3.6-27B-4bit (88%
  acceptance, 1.73x) is carded too — Qwen3.6 ships model_type=qwen3_5
  and works through the existing stack with zero code changes. MTP is
  skipped when logits processors are active (repetition penalty, bench
  EOS ban): accepted drafts commit from raw verifier logits, so
  processor-aware verification is required first (tracked follow-up). Known property: on hybrid (GDN) models, MTP greedy output is
  semantically greedy but not guaranteed byte-identical to non-MTP decode —
  the batched verify/replay chunked-scan numerics drift the recurrent state
  and can flip near-tie tokens.

### Fixed

- MTP is explicitly single-node for now: distributed placements (any group
  size > 1) disengage speculation with a logged fallback. Pipeline sharding
  was already excluded; tensor-parallel would mechanically run but
  accept/reject decisions consume per-rank RNG and cross-rank lockstep is
  unvalidated — a divergent decision would silently corrupt every rank's
  cache. Distributed MTP (TP lockstep validation, then pipeline
  draft/verify) is the next workstream.
- MTP terminal responses (EOS / max-tokens break paths) now finalize the
  detokenizer exactly once before yielding, matching the non-MTP path —
  sentencepiece-backed tokenizers buffer partial byte sequences until
  finalize() and could drop the last token's tail bytes (#180 item 4;
  latent for current tiktoken-backed targets).
- MTP drafting consumed post-final-norm hidden states; the trunk accessor
  now returns pre-norm hiddens (what the heads were trained on) and folds
  the final norm into the head callable, keeping main-path logits
  unchanged. Also fixed the accept-path token-history divergence (a
  never-emitted sampled token entered logits-processor history — PR #191
  review finding) and the pure-KV reject path dropping the emitted main
  token from processor history.

- Boot-time auto-update for the Skulk service: the LaunchAgent now runs
  `git pull`, `uv sync`, and the dashboard build through a wrapper
  (`deployment/install/skulk-startup.sh`) before exec'ing skulk. Failures of
  the pull / sync steps are non-fatal (logged to
  `~/.skulk/logs/skulk.prep.log` and the service boots whatever revision is
  on disk); a missing `dashboard-react/dist/` is fatal because the API has
  no UI to serve. Toggle with `SKULK_AUTO_UPDATE=0` in `~/.skulk/skulk.env`.
- Operator-editable env file at `~/.skulk/skulk.env`, copied from
  `deployment/install/skulk.env.example` on first install and never
  overwritten on re-run. Surfaces `SKULK_LIBP2P_NAMESPACE`,
  `SKULK_VERBOSITY`, `PYTHONUNBUFFERED`, debug toggles, and external-logging
  knobs without requiring a plist edit.
- Separate `foundation.foxlight.skulk-vector` LaunchAgent that runs Vector
  as its own process (via `deployment/install/vector-startup.sh`). Vector
  tails the captured `~/.skulk/logs/skulk.stdout.log` instead of piping
  through Skulk's process, so a slow VictoriaLogs sink can no longer
  backpressure inference threads. Opt out with `--no-vector` on the
  installer.
- `SKULK_LOGGING_EXTERNAL=1` mode in `exo.shared.logging`: structured JSON
  goes to stdout for an external shipper to consume, and Skulk does not
  spawn its own internal Vector subprocess. The launchd installer turns
  this on by default. JSON sink is now `enqueue=True` so log producers are
  decoupled from the sink's I/O.
- New operator guide at `website/docs/external-logging.md` covering the
  full Vector + VictoriaLogs + Grafana stack: central-host install, per-node
  configuration, JSON schema, and troubleshooting.

### Fixed

- Phase 1 MTP speculative decoding repaired on the post-ladder stack (it had
  never been validated end-to-end): the GDN softplus patch no longer probes
  foreign lazy modules (transformers 5.10 resolved a `compute_g` probe into
  an aria image-processing import requiring torchvision, crashing every GDN
  runner at startup); tied-embedding models (Qwen3.5 small variants) locate
  their output head via `embed_tokens.as_linear`; the MTP loop feeds the
  trunk correctly-shaped token batches; rejects snapshot/restore SSM
  (ArraysCache) state instead of zeroing it — the bug that degenerated
  hybrid-model output; and `mtp.safetensors` sidecars resolve via a new
  `build_sidecar_path` (sidecar repos have no `config.json`, so the model
  resolver rejected them and MTP silently never engaged). Verified
  end-to-end: greedy parity is byte-exact with MTP on vs off
  (Qwen3.5-2B-4bit + the FoxlightAI bf16 sidecar). Draft acceptance is
  currently 0% — the Phase 1 head intentionally omits the sidecar's
  transformer block (Phase 2); tracked separately.

### Changed

- mlx-vlm 0.5.0 → 0.6.1 (Gemma 4 MTP initiative Phase B). 0.6.1 ships the
  speculative-drafter catalog Phase C consumes (`gemma4_assistant`,
  `gemma4_unified_assistant`, `gemma4_dflash` — plus upstream
  `qwen3_5_mtp` and `deepseek_v4_mtp` drafters relevant to #194 and the
  DeepSeek sidecar path). All Skulk touchpoints verified against 0.6.1
  (prompt_utils, load_image_processor, dynamic `mlx_vlm.models.*` imports);
  dependency floors were already satisfied by the #188/#190 ladder.

- Web framework migrated to starlette 1.x (1.2.1) / fastapi 0.136, unified
  across darwin and linux. Test code uses `httpx2` for starlette's
  `TestClient` (the httpx-backed client is deprecated in 1.x); production
  HTTP-client code remains on `httpx`. This unblocks the mlx-vlm 0.6.x bump
  (which floors `starlette>=1.0.1`).
- MLX dependency version ladder (darwin): `mlx` 0.31.1 → 0.31.2, `mlx-vlm`
  0.4.4 → 0.5.0, `transformers` cap lifted to `>=5.5,<6`, and the Foxlight
  `mlx-lm` fork reconciled onto upstream v0.31.3 (`0.31.3.post1`, rev
  `e2f7ddcd`). The fork now carries only two non-upstream fixes (ArraysCache
  leak, DeepSeek-V3.2 lightning-indexer batch>1); float32 logprobs, GDN
  precision, and left-padding eval are absorbed upstream. mlx-vlm 0.5.0
  brings the `gemma4_assistant` drafter as a maintained dependency. (The
  ladder initially held `starlette<1.0`; the starlette 1.x migration above
  landed as its own change and removed that constraint.)
- Distributed/prefix-cache slow tests now select their model by available
  GPU working-set size: GPT-OSS-20B on machines that fit it, otherwise
  Llama-3.2-1B (override with `SKULK_TEST_DISTRIBUTED_MODEL`). Previously
  the hardcoded 20B memory-exhausted 16 GB machines.
- `test_batch_generate` B=1 vs B=2 equivalence is now teacher-forced with a
  relative logit tolerance instead of bit-exactness. Root cause of the
  divergence: M5-class Neural Accelerators run float32 GEMM (batch ≥ 2) at
  TF32-style reduced precision by default while GEMV (batch 1) stays full
  fp32 (ml-explore/mlx#3534; `MLX_ENABLE_TF32=0` opts out and restores
  bit-exactness). The test asserts under default precision — what
  production runs.
- The `opt_batch_gen` top-logprobs precompute patch is version-gated: it
  no-ops with a warning on mlx-lm ≥ 0.31.3 (BatchGenerator split) and
  `extract_top_logprobs` falls back to its synchronous path. Re-port
  tracked in #187.
- Two Vector configs now exist for the two transport modes:
  `deployment/logging/vector.yaml` keeps the original `stdin` source for
  the in-process subprocess shipper (used by Linux systemd installs and
  macOS `--no-vector` installs); `deployment/logging/vector-external.yaml`
  carries a `file` source tailing `~/.skulk/logs/skulk.stdout.log` plus a
  `remap` transform that drops non-JSON lines, used by the launchd
  `skulk-vector` agent.
- `deployment/install/install-launchd.sh` now installs both agents by
  default, manages `~/.skulk/skulk.env` (auto-flipping
  `SKULK_LOGGING_EXTERNAL` to match the chosen mode on `--no-vector`),
  supports `--no-vector` and `--uninstall`, drops `bash -lc` from the
  plist (so repo paths with spaces work), and produces a more useful
  post-install summary.
- `deployment/install/install-systemd.sh` and
  `deployment/systemd/skulk.service` now use the same wrapper +
  `~/.skulk/skulk.env` integration as macOS, with
  `EnvironmentFile=-%h/.skulk/skulk.env` so the unit picks up env-file
  knobs.
- `website/docs/run-skulk-as-a-service.md` updated for the auto-update,
  env-file customization, and Vector agent flow.

## [1.1.0] - 2026-05-03

### Added

- Headless-resilience deployment kit: a systemd user unit
  (`deployment/systemd/skulk.service`) with `Restart=on-failure` plus
  start-limit backoff, a macOS LaunchAgent
  (`deployment/launchd/foundation.foxlight.skulk.plist`) with conditional
  `KeepAlive`, and one-shot installers for each
  (`deployment/install/install-systemd.sh`, `install-launchd.sh`). The
  Linux installer enables user lingering so headless boxes autostart Skulk
  across reboots without an active login session.
- Startup port preflight (`exo.startup_recovery.preflight_api_port`) runs
  before component boot and exits with `EX_TEMPFAIL` (75) when the API
  port is held by a previous instance, so the service supervisor can
  retry with backoff instead of producing a confusing bind error mid-run.
- Tailscale connectivity layer (`exo.connectivity.tailscale`): detection via
  `tailscale status --json`, `TailscaleConnectivityConfig` in `skulk.yaml`,
  `GET /v1/connectivity/tailscale` API endpoint, and a status row in the
  dashboard's Node tab Runtime section. Cluster nodes can now span multiple
  physical networks over Tailscale (or Headscale).
- Operator panel — a mobile-first `/operator` route in the dashboard for
  remote cluster control: cluster-wide memory/GPU/temperature summary, per-node
  health cards, and a tap-twice-to-confirm node restart button that calls
  `POST /admin/restart`.
- `copyToClipboard()` helper (`dashboard-react/src/utils/clipboard.ts`) with
  a `document.execCommand` fallback so copy affordances work over plain HTTP,
  not just `localhost` or HTTPS.
- Operator-facing "Run Skulk as a service" guide at
  `website/docs/run-skulk-as-a-service.md` — quickstart-first,
  copy-paste install per platform, day-to-day operations table, reboot
  verification, troubleshooting, uninstall, and an advanced
  system-level systemd variant for niche server setups.
- Tailscale setup and troubleshooting guide at `website/docs/tailscale.md`.

### Fixed

- Dashboard `crypto.randomUUID()` replaced with the `uuid` npm package so chat
  session IDs generate correctly over plain HTTP (secure-context restriction).
- `StartLimitBurst` / `StartLimitIntervalSec` moved from `[Service]` to `[Unit]`
  in `skulk.service` — these directives are silently ignored in `[Service]`.
- API port preflight now gated behind `spawn_api` so `--no-api` worker nodes
  don't fail a port check they'll never bind.
- macOS log directory corrected to `~/.skulk/logs` in both the installer script
  and the ops-table in the user guide (was incorrectly `~/.cache/skulk/logs`).
- Tailscale status fields serialized as camelCase (`selfIp`, `dnsName`) to
  match FastAPI's default `by_alias=True` encoding; dashboard hook updated to
  match.
- Removed the unwanted selected-bar highlight (blue stroke) from the trace
  waterfall renderer.

## [1.0.3] - 2026-05-02

### Added

- Per-placement node exclusion. `POST /place_instance` accepts an optional
  `excluded_nodes` array; the master's planner treats those nodes as if
  absent from the topology when scoring candidate cycles for that single
  placement. Already-running instances on the listed nodes are unaffected.
  The dashboard's placement modal exposes click-to-toggle pills under
  "Available Nodes" so operators can mark exclusions before launch.
- `excluded_node_ids` query parameter on `GET /instance/previews` so the
  preview endpoint produces previews against the post-exclusion topology
  and the dashboard's cluster preview reflects the operator's intent
  pre-launch.
- Observability surface consolidation under one panel: Live (cluster health
  + cross-rank flight-recorder timeline + tracing toggle), Node (per-node
  diagnostics with a node selector that defaults to the master), Traces
  (saved-trace browser with inline filtering, expandable rows, native
  waterfall renderer; legacy traces page deleted).
- API trace janitor — hourly background task that drops saved trace files
  older than `tracing.retention_days` (default 3 days; configurable via
  `skulk.yaml`).
- New theme tokens: `errorFill` / `errorOnFill` / `warningFill` /
  `warningOnFill` (palette-independent solid-callout colors for
  iconography) and `errorOnSurface` / `warningOnSurface` (palette-aware
  text colors for callout body copy).

### Changed

- Snapshot bootstrap for follower recovery so newer nodes can hydrate
  cluster state from a master-published snapshot and replay only the retained
  tail instead of rebuilding from event `0`.
- Bounded live master replay retention so long-lived sessions no longer
  need to grow the active `events.bin` without limit.
- Dashboard state migrated from Zustand to Redux Toolkit + RTK Query. Same
  shapes, same persistence, native dedup / polling / cache invalidation.

### Docs

- Architecture documentation overhauled: `architecture.md` (narrative)
  and `architecture-reference.md` (dense fact-sheet) now both exist and
  are kept in sync as architectural shape changes land.
- Documented the rollout caveat for snapshot bootstrap plus bounded retention:
  mixed-version clusters are acceptable during upgrade, but all nodes should be
  upgraded before operators rely on compacted replay history as the steady
  state.

## [1.0.2] - 2026-04-19

### Added

- Explicit runtime notes and capability metadata for DeepSeek V3.2 trusted quantizations.
- Public model-behavior documentation for DeepSeek V3.2.
- A first release-notes workflow for Skulk, including this changelog and public docs release pages.

### Changed

- Hardened model-by-model capability handling across Gemma 4, Nemotron, Qwen 3.5, GPT-OSS, Llama Nemotron Nano, and DeepSeek V3.2 so more model behavior now comes from explicit cards and normalized runtime contracts instead of family-only fallbacks.
- Clarified the macOS build contract: `uv` is the canonical runtime path, and Nix is now documented as the reproducible tooling and validation path rather than a hidden alternate MLX runtime.
- Unified the Darwin Nix environment with the `uv` runtime contract by removing the stale MLX source-build override that no longer matched Skulk's official `mlx` + `mlx-metal` wheel path.
- Updated the README logo asset and related top-level presentation.

### Fixed

- Drove branch-wide `basedpyright` to zero with production-code fixes and tighter tests instead of suppressions.
- Fixed multiple MLX runtime and test-contract issues uncovered during the strict-typing cleanup, including native vision wrapper behavior, parser typing, warmup-path narrowing, and realistic test doubles.
- Fixed download re-download behavior when the model search path has not yet been configured at coordinator startup.
- Restored Python/Rust constructor compatibility for `NetworkingHandle` by giving the Python-facing API the same default bootstrap/listen behavior used elsewhere in the stack.

### Docs

- Updated contributor and agent guidance so build, formatting, and release-note expectations are explicit.
- Added dedicated build/runtime documentation describing how `uv` and Nix align and where they intentionally differ.
