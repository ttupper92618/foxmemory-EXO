---
id: api-guide
title: Skulk API
sidebar_position: 2
---

<!-- Copyright 2025 Foxlight Foundation -->

Skulk serves an API at `http://localhost:52415`.

That API has two jobs:

- compatibility endpoints for tools that already speak OpenAI, Claude, or Ollama-style APIs
- Skulk-specific control endpoints for placement, downloads, config, tracing, and model-store workflows

A model must be placed and running before chat requests for it succeed; calling
`/v1/chat/completions` for an unplaced model returns a 404 `No instance found`.
Text-generation endpoints require the mounted card to declare `TextGeneration`;
targeting a TTS-only or STT-only model returns **400 Bad Request** before any
runner command is dispatched. This applies to Chat Completions, Responses,
Claude, Ollama chat/generate, and benchmark adapters through their shared
admission path.
The [First Success Flow](#first-success-flow) below walks from placement to first
token.

## Quick Navigation

- First working request: [First Success Flow](#first-success-flow)
- OpenAI-compatible chat: [OpenAI Chat Completions](#openai-chat-completions)
- OpenAI Responses format: [OpenAI Responses API](#openai-responses-api)
- OpenAI embeddings: [OpenAI Embeddings API](#openai-embeddings-api)
- OpenAI text-to-speech: [OpenAI Audio Speech API](#openai-audio-speech-api)
- Image generation: [Image Generation and Editing](#image-generation-and-editing)
- Claude format: [Claude Messages API](#claude-messages-api)
- Ollama compatibility: [Ollama API](#ollama-api)
- Placement and launch: [Placement and Instance Management](#placement-and-instance-management)
- Store and config: [Model Store Endpoints](#model-store-endpoints) and [Configuration Endpoints](#configuration-endpoints)
- Debugging: [State, Events, and Tracing](#state-events-and-tracing)
- Pair a device: [Operator Device Pairing](#operator-device-pairing)

## First Success Flow

### 1. Start Skulk

Packaged users choose **Start Skulk** in the desktop app. From a source
checkout, run:

```bash
uv run skulk
```

### 2. Preview placements

```bash
curl "http://localhost:52415/instance/previews?model_id=mlx-community/Llama-3.2-1B-Instruct-4bit"
```

This shows what Skulk can actually place on the current node or cluster.

### 3. Launch a placement

```bash
curl -X POST http://localhost:52415/place_instance \
  -H 'Content-Type: application/json' \
  -d '{
    "model_id": "mlx-community/Llama-3.2-1B-Instruct-4bit",
    "sharding": "Pipeline",
    "instance_meta": "MlxRing",
    "min_nodes": 1
  }'
```

### 4. Send a chat request

```bash
curl -X POST http://localhost:52415/v1/chat/completions \
  -H 'Content-Type: application/json' \
  -d '{
    "model": "mlx-community/Llama-3.2-1B-Instruct-4bit",
    "messages": [{"role": "user", "content": "Hello from Skulk"}]
  }'
```

If this fails with `404 No instance found for model ...`, the placement is not ready yet or never launched successfully.

## Endpoint Overview

### Compatibility APIs

- `POST /v1/chat/completions`
- `POST /v1/responses`
- `POST /v1/embeddings`
- `POST /v1/audio/speech`
- `POST /v1/audio/transcriptions`
- `POST /v1/audio/translations`
- `GET /v1/audio/voices`
- `WS /v1/realtime`
- `WS /v1/fabric/chains/speech`
- `POST /v1/messages`
- `POST /v1/cancel/{command_id}`
- `POST /ollama/api/chat`
- `POST /ollama/api/generate`
- `GET /ollama/api/tags`
- `POST /ollama/api/show`
- `GET /ollama/api/ps`
- `GET /ollama/api/version`

The Ollama group also serves alias paths (`/ollama/api/api/...`,
`/ollama/api/v1/...`, and `HEAD` version probes); see
[Ollama API](#ollama-api).

### Images

- `POST /v1/images/generations`
- `POST /v1/images/edits`
- `GET /images`
- `GET /images/{image_id}`

### Benchmarking

- `POST /bench/chat/completions`
- `POST /bench/images/generations`
- `POST /bench/images/edits`

### Skulk Control APIs

- `GET /v1/models`
- `GET /models`
- `POST /v1/tools/web_search`
- `POST /v1/tools/open_url`
- `POST /v1/tools/extract_page`
- `GET /models/search`
- `POST /models/add`
- `POST /models/add-card`
- `DELETE /models/custom/{model_id}`
- `POST /place_instance`
- `POST /instance`
- `GET /instance/placement`
- `GET /instance/previews`
- `GET /instance/{instance_id}`
- `DELETE /instance/{instance_id}`
- `GET /state`
- `GET /events`
- `POST /download/start`
- `DELETE /download/{node_id}/{model_id}`
- `GET /config`
- `PUT /config`
- `GET /store/health`
- `GET /store/registry`
- `GET /store/downloads`
- `POST /store/models/{model_id}/download`
- `DELETE /store/models/{model_id}/download`
- `GET /store/models/{model_id}/download/status`
- `DELETE /store/models/{model_id}`
- `POST /store/purge-staging`
- `GET /store/storage`
- `POST /store/models/{model_id}/optimize`
- `GET /store/models/{model_id}/optimize/status`
- `GET /filesystem/browse`
- `GET /node/identity`
- `GET /node_id`
- `POST /admin/restart`
- `GET /onboarding`
- `POST /onboarding`
- `GET /v1/tracing`
- `PUT /v1/tracing`
- `GET /v1/telemetry/preview`
- `GET /v1/traces`
- `GET /v1/traces/cluster`
- `POST /v1/traces/delete`
- `GET /v1/traces/{task_id}`
- `GET /v1/traces/{task_id}/stats`
- `GET /v1/traces/{task_id}/raw`
- `GET /v1/traces/cluster/{task_id}`
- `GET /v1/traces/cluster/{task_id}/stats`
- `GET /v1/traces/cluster/{task_id}/raw`
- `GET /v1/diagnostics/node`
- `GET /v1/diagnostics/telemetry`
- `GET /v1/diagnostics/performance-envelopes`
- `GET /v1/diagnostics/performance-envelopes/cluster`
- `POST /v1/diagnostics/node/capture`
- `POST /v1/diagnostics/node/runners/{runner_id}/cancel`
- `GET /v1/diagnostics/cluster`
- `GET /v1/diagnostics/cluster/timeline`
- `GET /v1/diagnostics/cluster/{node_id}`
- `POST /v1/diagnostics/cluster/{node_id}/capture`
- `POST /v1/diagnostics/cluster/{node_id}/runners/{runner_id}/cancel`
- `GET /v1/capabilities`
- `POST /v1/capabilities/call`
- `POST /v1/capabilities/stream`
- `POST /v1/capabilities/stream/cancel`

### Operator Authentication

- `POST /v1/auth/pairing-sessions/challenge`
- `POST /v1/auth/pairing-sessions/exchange`
- `POST /v1/auth/token`
- `GET /v1/auth/devices`
- `DELETE /v1/auth/devices/{device_id}`

The node diagnostics bundle includes the node's own Tailscale state
(`tailscale`: running flag, tailnet IP, hostname, MagicDNS name), probed on
the node the bundle describes, so the per-node cluster endpoint reports the
selected node's tailnet identity rather than whichever node served the HTTP
request. The probe is best-effort: a node without a working `tailscale` CLI
reports `running: false`, and `null` marks only an unexpected probe failure.

For the full interactive reference with request/response schemas, see the [API Reference](/api/skulk-api).

## Operator Device Pairing

Operator pairing is explicitly started on the host that will act as the
designated remote gateway. It does not expose an HTTP endpoint that creates
pairing sessions:

```bash
uv run skulk operator configure-relay \
  --provisioning-file /protected/path/relay-v1.json \
  --operator-api-port 52417 \
  --cluster-name "Cluster"
```

The generated provisioning document is supplied by the Foxlight relay service
and contains `version`, `appWebsocketUrl`, `gatewayWebsocketUrl`,
`routingLocator`, distinct `appCarrierCredential` and
`gatewayCarrierCredential` values, and `laneCount`. Treat the file as a secret:
it contains both outer carrier roles. Skulk validates exact fixed carrier paths,
requires WSS except for loopback development, stores the route and credentials
inside the encrypted authority journal, generates an owner-only pinned TLS
identity, and refuses silent replacement. Restart Skulk after initial
configuration so the designated gateway opens its bounded outbound lane pool.
The operator listener defaults to loopback port `52417`, separate from Skulk's
default `52416` fabric transport.

```bash
uv run skulk operator pair \
  --cluster-name "Cluster"
```

The command initializes the gateway's encrypted local authority store when
needed, creates one high-entropy session that expires after five minutes, and
prints a terminal QR code plus the exact `skulk://pair?...` fallback payload.
When relay access is configured, the version-two QR uses it by default and
contains cluster identity, fingerprint, the single-use nonce and expiry, plus
the app-role outer carrier locator/credential and pinned inner-TLS material
needed to reach these same pairing routes. It never contains a canonical access
or refresh credential, or the gateway-role carrier credential. Treat the QR as
host-authorized pairing material and do not publish it. The app-role carrier
credential only admits an opaque relay lane; the five-minute nonce and device
proof still gate credential issuance at Skulk.

Version-two packages use bounded compact JSON compressed with zlib and carried
in the QR's single `z` query parameter. Skulk rejects a package before
persisting its session if it would exceed the terminal-scannable budget.
Version-one direct packages retain the uncompressed `payload` shape.

The legacy command remains five-minute and single-use for compatibility. For a
reusable, revocable version-three invitation, opt in with bounded host flags:

```bash
uv run skulk operator pair \
  --valid-for 90d \
  --max-pairings 10 \
  --qr-output review.png
```

`--valid-for` accepts a positive integer followed by `m`, `h`, or `d`, from one
minute through 90 days. Invitation mode permits ten successful pairings by
default and accepts an explicit limit from one through twenty. It creates a
separate five-minute attempt for every scan, so concurrent devices do not share
or replace challenges. At most ten attempts may be live and one hundred may be
issued over an invitation's lifetime. Only successful credential issuance
counts against the pairing limit.

The compressed version-three package adds a public invitation ID, issue time,
expiry, and pairing limit. It may carry the same app-role relay admission and
pinned inner-TLS material as version two, but never a canonical Skulk access or
refresh credential. Treat the QR and optional owner-only PNG as bearer secrets.
The PNG writer uses owner-only permissions and refuses to overwrite a path.

Invitation management remains local to the designated host. Headless operators
can use the CLI:

```bash
uv run skulk operator invitations list
uv run skulk operator invitations revoke <invitation-id>
```

Listing exposes only ID, creation and expiry times, usage, active-attempt
count, and safe state; it never prints the nonce. Revocation blocks new and
unfinished attempts but does not disconnect devices that already paired.
Revoke those devices through the authenticated device-management API.

The dashboard exposes the same authority operation under **Settings →
Pairing**. Operators choose a lifetime and device limit, generate a branded QR,
and may download or revoke it. The bearer QR remains in mounted browser memory
for five minutes and then the section resets; this display timeout does not
shorten a longer invitation. Safe invitation status remains visible without
the nonce or pairing code.

`--exchange-url https://gateway.example.invalid` remains an optional direct
LAN/Tailscale path and is required only before relay provisioning. Remote
exchange URLs must use HTTPS; cleartext HTTP is accepted only for loopback
development URLs. A relay-configured package includes both the protected inner
origin and relay bootstrap material, and the app prefers the relay path.

### Create a dashboard pairing invitation

**POST** `/v1/auth/pairing-invitations`

Parameters:

- JSON body `validForSeconds` (required): whole-second lifetime from 60 through
  7,776,000 seconds (90 days).
- JSON body `maxPairings` (required): successful device limit from 1 through
  20.
- Header `X-Skulk-Dashboard: pairing-v1` (required): explicit dashboard
  request marker.

Behavior:

- requires a loopback or Tailscale socket peer, an exact same-origin browser
  `Origin` or `Referer`, and a loopback, MagicDNS, `*.ts.net`, or literal
  Tailscale dashboard host;
- accepts Tailscale's `100.64.0.0/10` IPv4 and
  `fd7a:115c:a1e0::/48` IPv6 address spaces only after the local Tailscale
  authority verifies the exact socket peer, rejects proxy forwarding headers,
  and remains unavailable through an ordinary LAN or unverified CGNAT
  connection;
- is available only when the dashboard is opened on the configured operator
  gateway through Tailscale or localhost;
- is explicitly unavailable through `OperatorGatewayAuthorization`, even to a
  valid fully scoped device;
- reuses `OperatorPairingService.create_invitation`, so the CLI and dashboard
  cannot diverge in identity, relay material, limits, or journal behavior;
- returns safe `invitation` status plus one `pairingCode` containing the secret
  `skulk://pair?z=...` bearer package;
- returns `Cache-Control: no-store, max-age=0` and `Pragma: no-cache`; clients
  must not persist the response, put it in URLs, logs, telemetry, or shared
  application state;
- returns an actionable `403` outside the direct Tailscale/localhost dashboard
  authority, an actionable `409` on a non-gateway or before relay
  configuration, `422` if the generated package exceeds the reliable QR
  budget, and `503` when the configured gateway identity is temporarily
  unavailable.

### List dashboard pairing invitations

**GET** `/v1/auth/pairing-invitations`

Parameters:

- Header `X-Skulk-Dashboard: pairing-v1` (required).

Behavior:

- applies the same loopback-or-Tailscale peer, exact same-origin, trusted-host,
  and no-forwarding boundary as creation;
- returns invitation ID, creation/expiry, successful and maximum pairings,
  active/total attempts, and state;
- never returns the invitation nonce, QR payload, carrier credentials, or
  canonical operator credentials;
- returns an actionable `403` outside the direct verified
  Tailscale/localhost dashboard authority and an actionable `409` on a
  non-gateway or before relay configuration.

### Revoke a dashboard pairing invitation

**DELETE** `/v1/auth/pairing-invitations/{invitation_id}`

Parameters:

- Path `invitation_id` (required): public invitation UUID.
- Header `X-Skulk-Dashboard: pairing-v1` (required).

Behavior:

- applies the same loopback-or-Tailscale peer, exact same-origin, trusted-host,
  and no-forwarding boundary;
- blocks new and unfinished attempts without revoking already paired devices;
- returns `204` after success or idempotent repeated revocation, `404` for an
  unknown invitation, an actionable `403` outside the direct verified
  Tailscale/localhost dashboard authority, and `409` on a non-gateway, before
  relay configuration, or for an unresolved concurrent authority transition.

### Create a device challenge

**POST** `/v1/auth/pairing-sessions/challenge`

Parameters:

- JSON body `nonce` (required): high-entropy capability from the QR package.
  It is deliberately excluded from the URL so normal request-path logs cannot
  retain it.
- JSON body `deviceName` (required): operator-visible device label, 1–80
  characters after whitespace normalization.
- JSON body `devicePublicKey` (required): unpadded URL-safe base64 containing
  one raw 32-byte Ed25519 public key.
- JSON body `invitationId` (optional): version-three invitation UUID. Omit it
  for legacy single-use packages.

Behavior:

- accepts only a host-created, available session or invitation;
- binds the proposed device key to a legacy session or a new independent
  five-minute invitation attempt;
- returns a random base64url `challenge`, attempt `expiresAt`, and an
  `attemptId` only for version-three invitations;
- returns `404` for an unknown nonce, `410` after expiry, `409` after another
  transition already used the session, `422` for an invalid public key, `429`
  with `Retry-After` when ten invitation attempts are already live, and `503`
  on an API node that has not been initialized as a gateway. Revoked,
  exhausted or expired invitations return `410`; the lifetime attempt ceiling
  returns `410` only when requesting another attempt, while already-issued
  attempts may still finish before their own expiry.

Legacy version-one and version-two packages sign the domain-separated message
defined by `pairing_signature_message` in `src/skulk/operator/pairing.py`:

```text
"skulk-device-pairing-v1\\0"
  || ASCII(clusterId) || "\\0"
  || ASCII(nonce) || "\\0"
  || ASCII(challenge)
```

Version-three invitations instead sign the distinct message defined by
`pairing_invitation_signature_message`:

```text
"skulk-device-pairing-v2\\0"
  || ASCII(clusterId) || "\\0"
  || ASCII(invitationId) || "\\0"
  || ASCII(nonce) || "\\0"
  || ASCII(attemptId) || "\\0"
  || ASCII(challenge)
```

Here `||` means byte concatenation, each quoted `\\0` is one NUL byte, and
UUIDs use their canonical lowercase hyphenated representation. The v3 domain
and both returned identifiers are mandatory: signing the legacy transcript for
an invitation fails proof verification. Clients should use the corresponding
shared helper when available rather than maintaining another copy of these
bytes.

### Exchange device proof

**POST** `/v1/auth/pairing-sessions/exchange`

Parameters:

- JSON body `nonce` (required): the same pairing capability. It is deliberately
  excluded from the URL so normal request-path logs cannot retain it.
- JSON body `signature` (required): unpadded URL-safe base64 Ed25519 signature
  produced by the challenged device key.
- JSON body `invitationId` and `attemptId` (optional as a pair): required for a
  version-three invitation and omitted together for legacy packages.

Behavior:

- verifies possession of the exact device key bound during the challenge;
- atomically consumes the legacy session or exact invitation attempt;
- returns a stable `deviceId`, the validated cluster identity, a 15-minute
  opaque access token, a 30-day rotating refresh token, their expiries, and the
  granted canonical API scopes;
- when relay access is configured, also returns one-time `remoteAccess` material:
  `transport=paired_websocket_v1`, app WebSocket URL, opaque route locator,
  app-role carrier credential, inner-TLS server name, and pinned gateway CA
  certificate. The gateway-role carrier credential is never returned;
- stores only encrypted session/device state and one-way token digests;
- never returns either credential again;
- returns `401` for an invalid proof, `404` for an unknown invitation or
  attempt, `409` for reuse or an out-of-order exchange, `410` when the session,
  attempt, or invitation is unavailable, and `503` on a non-designated node.

Together with refresh rotation, these are the complete pre-access-token HTTP
surface on the relay-only listener. That listener serves the existing canonical
Skulk app rather than a parallel mobile API: safe reads require `cluster:read`
or `models:read`, inference/WebSocket routes require `chat:write`, mutations
require `operations:write`, and device routes require `devices:manage`. The
existing local listener and dashboard remain unchanged for direct clients; only
the separate loopback TLS listener connected to the relay applies this bearer
boundary.

### Rotate operator credentials

**POST** `/v1/auth/token`

Parameters:

- JSON body `deviceId` (required): stable paired-device UUID returned by the
  exchange.
- JSON body `refreshToken` (required): current opaque rotating refresh
  credential.

Behavior:

- validates the device and current refresh-token digest;
- atomically invalidates both members of the previous token pair;
- returns a fresh 15-minute access token and 30-day refresh token, their
  expiries, and the device's unchanged scopes;
- returns `401` for an unknown, revoked, expired, or replayed credential and
  `409` if concurrent credential state changed.

The client must replace both stored credentials as one operation. A response
lost after the gateway commits rotation requires pairing again because replaying
the previous refresh token is intentionally rejected. Relay and pinned TLS
material are not repeated during refresh; the app retains them in platform
secure storage until it disconnects or pairs again.

### List paired devices

**GET** `/v1/auth/devices`

Parameters:

- `Authorization: Bearer <access-token>` (required): a valid credential with
  `devices:manage` scope.

Behavior:

- returns stable device IDs, display names, pairing times, refresh expiries,
  active/revoked state, and which row represents the caller;
- never returns device public keys, token digests, raw credentials, or pairing
  nonces;
- returns `401` for a missing, malformed, unknown, revoked, or expired bearer
  and `403` when the bearer lacks device-management scope.

### Revoke a paired device

**DELETE** `/v1/auth/devices/{device_id}`

Parameters:

- path `device_id` (required): stable paired-device UUID to revoke;
- `Authorization: Bearer <access-token>` (required): a valid credential with
  `devices:manage` scope.

Behavior:

- atomically clears the target's access and refresh digests and expiries;
- returns `204` after successful revocation and when an authorized caller
  repeats revocation for the same already-revoked target;
- makes both credentials unusable immediately;
- returns `401` for an invalid caller, `403` for insufficient scope, `404` for
  an unknown target device, and `409` if concurrent credential state changed.

## OpenAI Chat Completions

**POST** `/v1/chat/completions`

This is the main chat-generation endpoint for both text-only and multimodal
models.

Requests are validated before dispatch: an empty `messages` array or a
non-positive `max_tokens` returns **400 Bad Request** rather than being
accepted and failing during generation. (This applies across the Claude,
Ollama, and Responses wire formats too, which share the same dispatch path.)
The mounted model must also declare `TextGeneration`; speech-only cards return
**400 Bad Request** without affecting their speech runner.

### Context-length limits

Every placed instance has a usable context limit: the smaller of the model's
advertised context length and the number of KV-cache tokens that fit in memory
next to the model weights on the hosting node(s). Requests are admitted
against that limit instead of growing the KV cache until the node runs out of
memory:

- A `max_tokens` value that cannot fit in the limit at all returns
  **400 Bad Request** immediately (`context_length_exceeded: ...`).
- After tokenization on the serving instance, a prompt that fills the window,
  or a prompt plus an explicit `max_tokens` that exceeds the limit, is
  rejected with an OpenAI-style `invalid_request_error` whose message starts
  with `context_length_exceeded:`. The HTTP status is already committed when
  the rejection is computed on the serving node, so this arrives in the body:
  as the first SSE `data:` event for streaming requests, and as the response
  body for non-streaming ones.
- When `max_tokens` is omitted, the server default output budget is clamped to
  the remaining window, so generation ends with `finish_reason: "length"`
  instead of overrunning the context.

### OpenAI Python SDK Example

```python
from openai import OpenAI

client = OpenAI(
    base_url="http://localhost:52415/v1",
    api_key="unused",
)

response = client.chat.completions.create(
    model="mlx-community/Llama-3.2-1B-Instruct-4bit",
    messages=[{"role": "user", "content": "Hello!"}],
)
print(response.choices[0].message.content)
```

### Curl Example

```bash
curl -X POST http://localhost:52415/v1/chat/completions \
  -H 'Content-Type: application/json' \
  -d '{
    "model": "mlx-community/Llama-3.2-1B-Instruct-4bit",
    "messages": [{"role": "user", "content": "Hello!"}]
  }'
```

### Streaming Example

```python
stream = client.chat.completions.create(
    model="mlx-community/Llama-3.2-1B-Instruct-4bit",
    messages=[{"role": "user", "content": "Tell me a story"}],
    stream=True,
)
for chunk in stream:
    if chunk.choices and chunk.choices[0].delta.content:
        print(chunk.choices[0].delta.content, end="")
```

#### Streaming response shape

Streaming responses are Server-Sent Events. Each event is a `data:` line
carrying one chunk object, and the stream ends with a literal `data: [DONE]`
sentinel:

```
data: {"id":"...","object":"chat.completion.chunk","created":1787328699,
       "model":"mlx-community/Llama-3.2-1B-Instruct-4bit",
       "choices":[{"index":0,"delta":{"role":"assistant","content":"Once"},
                   "finish_reason":null}],"usage":null}

data: {"id":"...","object":"chat.completion.chunk", ... ,
       "choices":[{"index":0,"delta":{"content":" upon"},"finish_reason":null}]}

data: [DONE]
```

Two differences from the non-streaming response matter to clients:

- `object` is `chat.completion.chunk`, not `chat.completion`. Strict clients
  validate this discriminator and reject a stream that carries the
  non-streaming value.
- Each choice carries a `delta` holding only what is new, rather than a
  complete `message`.

Skulk also emits SSE comment lines, which begin with `:` and which a
compliant client ignores. These carry the command id at the start of the
stream, and generation statistics at the end when available. A client that
treats every non-blank line as data will need to skip them.

Reasoning models place thinking text on `delta.reasoning_content` rather than
`delta.content`, so a client that reads only `content` shows the answer
without the reasoning. Tool calls arrive as a frame whose `delta.tool_calls`
carries the accumulated call and whose `finish_reason` is `tool_calls`.

#### When generation fails mid-response

A request that reaches a serving instance has already committed its HTTP
status by the time generation runs, streaming or not, so a failure after that
point is reported in the body rather than by the status. It carries an
`error` object holding `message`, `type` and `code`, the same shape a request
rejected before generation returns:

- **Non-streaming**: the body is the error object instead of a completion.
  Check for an `error` key before reading `choices`.
- **Streaming**: a `data:` frame carrying the error object, followed by
  `data: [DONE]`. The stream always terminates with the sentinel, including
  when the task is cancelled or ends without completing, so a client can
  distinguish a finished turn from a dropped connection.

The status is committed early on purpose. It is what lets the server notice
that a caller has disconnected and stop the generation, rather than producing
tokens for a client that has gone away.

A response body is never empty, and a partial answer is never presented as a
complete one. A turn ends only when the model reports a finish reason, so a
request that produced nothing, or that produced text and then stopped without
one, returns an error object rather than zero bytes or a silently truncated
completion.

### Common Request Fields

| Field | Type | Notes |
|-------|------|-------|
| `model` | string | Required. Must match a placed and running model. |
| `messages` | array | Required. Supports `system`, `user`, `assistant`, `developer`, `tool`, `function`. |
| `stream` | boolean | Use `true` for SSE streaming. |
| `temperature` | number | Sampling temperature. |
| `top_p` | number | Nucleus sampling. |
| `top_k` | integer | Top-k sampling. |
| `min_p` | number | Minimum-probability threshold. |
| `max_tokens` | integer | Max generated tokens. When omitted, Skulk uses the shared default of 4096 generated tokens (`DEFAULT_MAX_OUTPUT_TOKENS`); operators can override it with `SKULK_MAX_OUTPUT_TOKENS` (or the legacy `SKULK_MAX_TOKENS`). The served llama.cpp engine uses this finite bound with its exact rendered input-token count to reserve shared KV capacity before generation; waiters enter that pool FIFO. |
| `stop` | string or array | Stop sequences. |
| `seed` | integer | Reproducibility helper. |
| `frequency_penalty` | number | Frequency penalty. |
| `presence_penalty` | number | Presence penalty. |
| `repetition_penalty` | number | Repetition penalty. |
| `repetition_context_size` | integer | Context window for repetition handling. |
| `logprobs` | boolean | Return token logprobs when supported. |
| `top_logprobs` | integer | Number of top logprobs to include. |
| `tools` | array | OpenAI-style tool definitions. |
| `tool_choice` | string or object | `auto`, `none`, or a specific tool selection. |
| `parallel_tool_calls` | boolean | Accepted for compatibility. |
| `enable_thinking` | boolean | Skulk extension for reasoning-capable models. |
| `reasoning_effort` | string | Reasoning hint when supported. |
| `response_format` | object | Accepted for compatibility, not strictly enforced. |
| `stream_options` | object | Includes `include_usage`. |
| `user` | string | Optional caller identifier. |

### Message Format

```json
{
  "role": "user",
  "content": "hello"
}
```

Assistant messages may include `tool_calls`.
Tool response messages should include `tool_call_id`.

User messages may also be sent as structured content parts. Skulk accepts
OpenAI-style image inputs for vision-capable models:

```json
{
  "role": "user",
  "content": [
    { "type": "text", "text": "What is in this image?" },
    {
      "type": "image_url",
      "image_url": { "url": "data:image/png;base64,..." }
    }
  ]
}
```

Notes:

- inline `data:` URLs are supported for image inputs
- Anthropic-compatible requests can also carry image content for multimodal models
- image understanding depends on the selected model exposing the `vision` capability
- request-scoped encoded image media is limited to 32 MiB; multipart image-edit
  uploads are read through the corresponding 24 MiB raw-image limit
- image bytes are sent only after authoritative task placement, directly to the
  selected worker rank or ranks; they are bounded, integrity-checked, and each
  rank must acknowledge task-owner verification before the transfer deadline
- image bytes are never written to the event log or replicated `State`

### Finish Reasons

| Value | Meaning |
|-------|---------|
| `stop` | Natural stop or stop sequence reached |
| `length` | `max_tokens` limit reached |
| `tool_calls` | Model is requesting a tool call |
| `content_filter` | Reserved for compatibility |
| `function_call` | Reserved for compatibility |
| `error` | Generation failed |

## Tool Use

Skulk supports OpenAI-style function calling.

```python
tools = [{
    "type": "function",
    "function": {
        "name": "get_weather",
        "description": "Get current weather for a city",
        "parameters": {
            "type": "object",
            "properties": {
                "location": {"type": "string"}
            },
            "required": ["location"]
        }
    }
}]

response = client.chat.completions.create(
    model="mlx-community/Qwen3.5-9B-4bit",
    messages=[{"role": "user", "content": "What is the weather in Paris?"}],
    tools=tools,
    tool_choice="auto",
)
```

Typical flow:

1. Send messages and tool definitions.
2. Inspect `finish_reason`.
3. If it is `tool_calls`, execute the tool in your app.
4. Send the tool result back as a `tool` message.
5. Request the final model response.

Two behaviors are worth knowing when you send `tools`.

**Only the tools you offer come back.** Some models reach for a built-in of
their own rather than one of yours: Llama answers some plain questions by
calling `print`, and gpt-oss has `python` and `browser`. A response that names
no tool you offered is returned as ordinary content with its normal
`finish_reason`, not as a `tool_calls` response, because you would have no
implementation to run. Check `finish_reason` rather than assuming a response is
a call.

**`tool_choice` means the same thing on every engine.** `"none"` removes the
tools from the request, which is the only way to guarantee the documented
behavior that the model does not call one: a model handed a tool and asked for
it will call it whatever the request said. Naming a single function narrows the
offered tools to that one, so the model cannot call a different tool than you
asked for. A name matching none of your tools rejects the request with a
`400`, on every engine, because your forced choice cannot be honored and any
answer would be a guess at what you meant; forcing a name while offering no
tools at all is rejected the same way.
`"auto"` and `"required"` pass through, and
`"required"` is a best-effort instruction on the in-process engines rather than
a guarantee, because forcing a call there would need constrained decoding.

**A JSON answer stays an answer.** Several model families write a tool call as
a bare JSON object, so a request that both offers tools and asks for JSON output
is ambiguous on the wire. Skulk resolves it in favor of the answer: text that
does not parse as a call to one of your tools is returned as content. A short
prefix of such a message is buffered while it could still be either reading,
and released the moment it is distinguishable, so a JSON answer streams
incrementally after a delay bounded by its first decisive key rather than
arriving in one piece.

## Thinking / Reasoning

Skulk supports reasoning-aware chat for compatible models.

```python
response = client.chat.completions.create(
    model="mlx-community/Qwen3.5-9B-4bit",
    messages=[{"role": "user", "content": "What is 127 * 43?"}],
    enable_thinking=True,
)

msg = response.choices[0].message
print(msg.reasoning_content)
print(msg.content)
```

Notes:

- `enable_thinking` is a Skulk extension.
- Reasoning support depends on model capabilities.
- Use `GET /v1/models` response `data[].resolved_capabilities` to decide whether a model supports thinking and whether clients should render a thinking toggle.
- Treat `resolved_capabilities` as the default tool-free request path; request-specific options such as tools can change prompt rendering and related resolved values for mixed-mode model families.
- Thinking-control semantics are model-aware:
  - if `supports_thinking_toggle` is `true`, send `enable_thinking=true` or `false` explicitly
  - if both `enable_thinking` and `reasoning_effort` are omitted for a model with a known toggleable capability profile, Skulk disables thinking using the profile's disabled effort
  - `reasoning_effort="none"` disables thinking for toggleable models
  - if a model does not support toggleable thinking, Skulk ignores explicit toggle overrides but still preserves explicit non-disabled reasoning-effort hints when the model family supports them

## Builtin Browser Tools

**POST** `/v1/tools/web_search`

Execute Skulk's generic `web_search` tool and return structured search results.

```bash
curl -X POST http://localhost:52415/v1/tools/web_search \
  -H 'Content-Type: application/json' \
  -d '{
    "query": "foxlight skulk distributed inference",
    "top_k": 5
  }'
```

Request fields:

| Field | Type | Notes |
|-------|------|-------|
| `query` | string | Required search query. |
| `top_k` | integer | Optional max results, `1` to `10`, default `5`. |

Response fields:

| Field | Type | Notes |
|-------|------|-------|
| `query` | string | Original search query. |
| `provider` | string | Search backend identifier. |
| `results` | array | Ordered search results with `title`, `url`, and `snippet`. |

This endpoint is designed for client-executed tool loops. GPT-OSS can request
`web_search`, the client can call this endpoint, then send the JSON result back
as a `tool` message.

**POST** `/v1/tools/open_url`

Fetch one HTTP or HTTPS URL, follow redirects, and return structured metadata.

```bash
curl -X POST http://localhost:52415/v1/tools/open_url \
  -H 'Content-Type: application/json' \
  -d '{
    "url": "https://example.com/article"
  }'
```

Request fields:

| Field | Type | Notes |
|-------|------|-------|
| `url` | string | Required absolute `http://` or `https://` URL. |

Response fields:

| Field | Type | Notes |
|-------|------|-------|
| `url` | string | Original requested URL. |
| `final_url` | string | Final URL after redirects. |
| `title` | string or null | Best-effort page title. |
| `status_code` | integer | Final HTTP status code. |
| `content_type` | string or null | Normalized response content type. |
| `provider` | string | Backend provider identifier. |

**POST** `/v1/tools/extract_page`

Fetch one HTTP or HTTPS URL and return bounded readable text extracted from the
response body.

```bash
curl -X POST http://localhost:52415/v1/tools/extract_page \
  -H 'Content-Type: application/json' \
  -d '{
    "url": "https://example.com/article",
    "max_chars": 12000
  }'
```

Request fields:

| Field | Type | Notes |
|-------|------|-------|
| `url` | string | Required absolute `http://` or `https://` URL. |
| `max_chars` | integer | Optional maximum characters, `500` to `50000`, default `12000`. |

Response fields:

| Field | Type | Notes |
|-------|------|-------|
| `url` | string | Original requested URL. |
| `final_url` | string | Final URL after redirects. |
| `title` | string or null | Best-effort page title. |
| `text` | string | Readable extracted text. |
| `truncated` | boolean | Whether the text was clipped to `max_chars`. |
| `provider` | string | Backend provider identifier. |

These browser-tool endpoints are designed for client-executed tool loops. In
dashboard chat, GPT-OSS can request `web_search`, `open_url`, or
`extract_page`; the dashboard executes the endpoint call, then sends the JSON
result back as a `tool` message.

## Structured Output

`response_format` is accepted for compatibility, but Skulk does not currently enforce strict JSON mode or JSON schema validation.

```python
response = client.chat.completions.create(
    model="mlx-community/Qwen3.5-9B-4bit",
    messages=[{"role": "user", "content": "Return valid JSON with three colors"}],
    response_format={"type": "json_object"},
)
```

For the best results, explicitly instruct the model to return valid JSON.

## OpenAI Responses API

**POST** `/v1/responses`

Use this when your client expects the OpenAI Responses format instead of Chat Completions.

```bash
curl -X POST http://localhost:52415/v1/responses \
  -H 'Content-Type: application/json' \
  -d '{
    "model": "mlx-community/Llama-3.2-1B-Instruct-4bit",
    "input": "Hello from the Responses API"
  }'
```

## OpenAI Embeddings API

**POST** `/v1/embeddings`

Generates embeddings with a mounted embedding model. The model must be placed
and running, and its card must declare `TextEmbedding`: a non-embedding model
returns **400 Bad Request**, and an unplaced model returns **404 No instance
found**. An alias absent from the authorized model catalog also returns **404**;
inference never discovers it from Hugging Face as a side effect.

```bash
curl -X POST http://localhost:52415/v1/embeddings \
  -H 'Content-Type: application/json' \
  -d '{
    "model": "BAAI/bge-small-en-v1.5",
    "input": ["Skulk connects devices into one cluster"]
  }'
```

Request fields:

| Field | Type | Notes |
|-------|------|-------|
| `model` | string | Required mounted embedding model id |
| `input` | string or array | Required text or list of texts; an empty list returns **400 Bad Request** |
| `encoding_format` | string | `float` (default) returns number arrays; `base64` returns each embedding as base64-encoded little-endian float32 bytes |
| `dimensions` | integer | Not supported. Any value returns **400 Bad Request**; embeddings come back at the model's native dimensionality |
| `user` | string | Optional caller identifier, accepted for compatibility |

The response is the OpenAI list shape: one `data[]` entry per input in input
order, the resolved `model`, and `usage` with prompt and total token counts.

## OpenAI Audio Speech API

**POST** `/v1/audio/speech`

Generates speech audio from a mounted text-to-speech model. The model must be
placed and running, and its resolved capabilities must include
`supports_speech_synthesis`. An alias absent from the authorized model catalog
returns **404** rather than triggering Hub discovery.

```bash
curl -X POST http://localhost:52415/v1/audio/speech \
  -H 'Content-Type: application/json' \
  --output speech.wav \
  -d '{
    "model": "mlx-community/kokoro-test",
    "input": "Hello from Skulk speech serving",
    "voice": "af_heart",
    "response_format": "wav"
  }'
```

Request fields:

| Field | Type | Notes |
|-------|------|-------|
| `model` | string | Required mounted TTS model id |
| `input` | string | Required text to synthesize |
| `voice` | string or null | Optional stable voice identifier. Accepted only when the mounted card declares a static `audio.voices` catalog; unknown names are rejected. Entries may be model-native speakers or checksummed reference profiles bundled with Skulk. When omitted, Skulk applies the card's `audio.default_voice` when declared. |
| `speed` | number or null | Optional positive speaking speed multiplier |
| `response_format` | string or null | Optional output format: `mp3`, `wav`, `flac`, `ogg`, `opus`, or raw `pcm`. When omitted or set to `null`, Skulk uses `mp3` for `stream=true`; otherwise it uses the mounted model card default when declared and falls back to `mp3`; supported values are constrained by the model card when declared |
| `stream` | boolean | Optional. When `true`, Skulk returns a chunked HTTP response and yields MP3 or raw PCM bytes as the speech runner emits them; accepted only when the mounted TTS card explicitly declares `audio.supports_streaming = true` and every routable instance of the requested model has a ready runner |
| `streaming_interval` | number or null | Optional positive model-specific streaming cadence hint, accepted only with `stream=true` |
| `instruct`, `lang_code` | string or null | Optional model-specific generation hints |
| `temperature`, `top_p`, `top_k`, `repetition_penalty` | number | Optional model-specific sampling controls |
| `max_tokens` | integer or null | Optional model-specific generation ceiling. An explicit value is preserved. When omitted and the mounted model explicitly declares this control, Skulk uses a 4096-token serving default instead of inheriting a potentially truncating upstream library default. Models that do not declare the control receive no injected keyword. |
| `seed` | integer or null | Optional unsigned 32-bit sampling seed. When supplied, the speech runner resets MLX sampling immediately before generation so identical model, voice, text, and sampling controls are reproducible. Omission preserves the upstream advancing random stream. |
| `reference_audio` | multipart file or null | Optional request-scoped voice-conditioning audio. Accepted only as a multipart upload for a mounted card declaring `audio.supports_reference_audio = true`; server-local paths are rejected |
| `reference_text` | string or null | Optional transcript of `reference_audio`; accepted only when the multipart upload is present |

The response body is raw audio bytes with a matching audio media type
(`audio/mpeg`, `audio/wav`, `audio/flac`, `audio/ogg`, `audio/opus`, or
`audio/pcm`).
For `stream=true`, the mounted TTS card must explicitly declare
`audio.supports_streaming = true`. The response format must currently
resolve to `mp3` or `pcm`; when a streaming request omits `response_format`, Skulk
requests `mp3` instead of the model card's non-streaming default. Skulk returns
`audio/mpeg` or `audio/pcm` with chunked HTTP bytes. Raw `pcm` is mono signed
16-bit little-endian audio; `X-Audio-Sample-Rate`, `X-Audio-Channels`, and
`X-Audio-Sample-Format` define its framing. Admission returns `503` if any routable
instance of the requested model lacks a ready runner. This is TTS output streaming, not a
realtime session: the request text is still a complete bounded input,
cancellation closes the command stream, and each chunk follows the mounted
model's generation cadence. The bundled Qwen3 TTS card declares MP3 and PCM streaming
support after live validation; Fish Audio and the other bundled speech cards
remain non-streaming. Streaming support is enabled card-by-card only when
the runtime can provide the encoder and the model has passed streaming
validation.

JSON requests remain text-only. To condition a supporting model with reference
audio, send the same scalar fields as multipart form values and include a
`reference_audio` file of at most 25 MiB. Skulk validates the mounted model and
audio metadata, pins the request to one ready single-host instance, and sends
the bytes over the node-addressed Zenoh data plane. Reference media is never
written to State or the event log, and the serving runner deletes its temporary
file when generation ends or fails. Reference-audio requests return **503
Service Unavailable** when the Zenoh data plane is unavailable; Skulk never
broadcasts private reference media through the gossipsub fallback.

Reference-capable bundled cards may also expose Skulk's packaged voice profiles
through the ordinary `voice` field. The worker resolves the selected identifier
to its checksummed local MP3 and exact transcript; those asset paths and bytes
never enter the command, State, or event log. This path does not require an
upload or Zenoh media transfer. A multipart `reference_audio` upload is an
explicit request-scoped override and cannot be combined with `voice`.

`streaming_interval` without `stream=true`, `reference_text` without a
multipart reference upload, a multipart upload combined with `voice`, and JSON
`reference_audio` path strings return **400 Bad Request**.

## Skulk Audio Voices API

**GET** `/v1/audio/voices?model=<model-id>`

Returns stable model-native and bundled-reference voice identifiers declared by
one mounted TTS model.
This is a Skulk extension, not an OpenAI compatibility route. The model must
declare `audio.supports_voice_listing = true`; otherwise Skulk returns **400 Bad
Request**.

```bash
curl 'http://localhost:52415/v1/audio/voices?model=org/tts-model'
```

The response is `{ "object": "list", "data": [...] }`. Each item contains the
voice `id`, display `name`, mounted `model`, a `kind` of `"builtin"` for a
model-native speaker or `"reference"` for a checksummed profile shipped with
Skulk, and an ordered `preferred_languages` array of BCP 47 tags when the model
card declares language preferences. This endpoint does not create or persist
user voice profiles.

## OpenAI Audio Transcriptions API

**POST** `/v1/audio/transcriptions`

Transcribes a multipart audio upload with a mounted speech-to-text model. The
model must be placed and running, and its resolved capabilities must include
`supports_transcription`.

```bash
curl -X POST http://localhost:52415/v1/audio/transcriptions \
  -F model=mlx-community/whisper-test \
  -F file=@sample.wav \
  -F response_format=verbose_json
```

Request fields:

| Field | Type | Notes |
|-------|------|-------|
| `file` | file | Required audio upload. Common WAV, MP3, FLAC, OGG, Opus, WebM, MP4/M4A containers are accepted up to 25 MiB |
| `model` | string | Required mounted STT model id |
| `language` | string or null | Optional input language hint |
| `prompt`, `context`, `text` | string or null | Optional model-specific transcription context |
| `response_format` | string | Optional output format: `json`, `text`, `verbose_json`, `srt`, `vtt`, or `ndjson`; default `json` |
| `temperature`, `max_tokens`, `chunk_duration`, `frame_threshold`, `prefill_step_size` | number or integer | Optional model-specific generation controls passed through only when the runner supports them |
| `word_timestamps` | boolean | Optional request for word timestamp metadata when supported |
| `timestamp_granularities` | string | Optional comma-separated or JSON-list timestamp granularity hints |
| `stream` | boolean | Optional. Requires a mounted card declaring `audio.supports_streaming = true` and ready runners. Returns typed SSE events by default; explicit `response_format=ndjson` retains progressive NDJSON framing. |

Response formats:

| Format | Media type | Shape |
|--------|------------|-------|
| `json` | `application/json` | `{ "text": "..." }` |
| `text` | `text/plain` | Plain transcript text |
| `verbose_json` | `application/json` | Transcript text plus language and segment metadata when the model returns it |
| `srt` | `application/x-subrip` | Subtitle output from model segments, with a zero-length fallback segment when timestamps are absent |
| `vtt` | `text/vtt` | WebVTT subtitle output from model segments |
| `ndjson` | `application/x-ndjson` | One JSON line per transcription chunk |

The endpoint never accepts server-local file paths. The API retains the bounded
multipart upload until the master selects the authoritative task placement,
then sends raw audio frames directly to the selected worker over
`SPEECH_MEDIA`. Only control-sized metadata and the task lifecycle enter the
ordered event log. The worker verifies the owner, frame count, and SHA-256
before injecting the payload into the serving runner; the runner writes a
temporary local audio file only while inference executes. With `stream=true`,
supported models yield their actual decoded text deltas. The default
`text/event-stream` response emits typed
`transcription.delta`, `transcription.completed`, `transcription.usage`, and
`transcription.error` events. Disconnecting before a terminal event cancels the
core command and releases its bounded output queue. An explicit
`response_format=ndjson` streams the existing per-chunk JSON shape one line at
a time. Cards without proven streaming support fail before response headers.
On Zenoh, upload frames are addressed to the selected worker. The gossipsub
fallback broadcasts target-tagged frames across the trusted cluster fabric;
non-target workers discard them before speech assembly.

## OpenAI Audio Translations API

**POST** `/v1/audio/translations`

Experimentally translates a multipart speech upload into English. The request
uses the same 25 MiB bounded upload path and response formats as transcription.
The mounted card must declare `audio.supports_translation = true`.

```bash
curl -X POST http://localhost:52415/v1/audio/translations \
  -F model=org/canary-model \
  -F file=@sample-fr.wav \
  -F language=fr \
  -F response_format=json
```

| Field | Type | Notes |
|-------|------|-------|
| `file` | file | Required bounded audio upload |
| `model` | string | Required mounted translation-capable STT model id |
| `language` | string or null | Optional source-language hint; required by the bundled Canary model |
| `prompt` | string or null | Optional model-specific translation context |
| `response_format` | string | `json`, `text`, `verbose_json`, `srt`, `vtt`, or `ndjson`; default `json` |
| `temperature` | number or null | Optional model-specific sampling temperature |

Translation target is English. The only gates are model truth and instance
availability: the mounted card must declare `audio.supports_translation =
true`, matching every other speech endpoint. Skulk maps the generic request to
model-family arguments inside the speech runner. The bundled
`CogniSoftOrg/canary-1b-v2-mlx-bf16` card is the initial supported model;
requests for that model return **400 Bad Request** when `language` is omitted.
Its upstream CC-BY-4.0 terms and NVIDIA attribution continue to apply.

## Claude Messages API

**POST** `/v1/messages`

Use this when your client expects Anthropic-style request and response shapes.

```bash
curl -X POST http://localhost:52415/v1/messages \
  -H 'Content-Type: application/json' \
  -d '{
    "model": "mlx-community/Llama-3.2-1B-Instruct-4bit",
    "messages": [{"role": "user", "content": "Hello"}],
    "max_tokens": 512
  }'
```

## Ollama API

Skulk supports several Ollama-compatible endpoints so tools like OpenWebUI can connect with minimal glue code.

### Chat

```bash
curl -X POST http://localhost:52415/ollama/api/chat \
  -H 'Content-Type: application/json' \
  -d '{
    "model": "mlx-community/Llama-3.2-1B-Instruct-4bit",
    "messages": [{"role": "user", "content": "Hello"}]
  }'
```

### Generate

```bash
curl -X POST http://localhost:52415/ollama/api/generate \
  -H 'Content-Type: application/json' \
  -d '{
    "model": "mlx-community/Llama-3.2-1B-Instruct-4bit",
    "prompt": "Write a haiku about foxes"
  }'
```

### List models

```bash
curl http://localhost:52415/ollama/api/tags
```

### Show model details

```bash
curl -X POST http://localhost:52415/ollama/api/show \
  -H 'Content-Type: application/json' \
  -d '{"name": "mlx-community/Llama-3.2-1B-Instruct-4bit"}'
```

### Alias routes

Ollama clients differ in how they join a configured base URL with API paths, so
Skulk also serves alias routes that map onto the same handlers:

- `POST /ollama/api/api/chat` and `POST /ollama/api/v1/chat` alias
  `POST /ollama/api/chat`
- `GET /ollama/api/api/tags` and `GET /ollama/api/v1/tags` alias
  `GET /ollama/api/tags`
- `HEAD /ollama/` and `HEAD /ollama/api/version` answer the version probe some
  clients send before their first real request

## Image Generation and Editing

Skulk serves OpenAI-style image generation and editing from placed image
models (for example the bundled FLUX cards).

Availability note: these routes are always registered, but they return
**404 No instance found** until an instance of the requested image model is
placed and running. Image model cards are hidden from the model catalog
(`GET /v1/models`, placement previews, and the dashboard) unless the node runs
with `SKULK_ENABLE_IMAGE_MODELS=true`, so in practice serving image models
requires setting that environment variable before launching one.

### Generate images

**POST** `/v1/images/generations`

The requested image model must already exist in the authorized catalog and be
placed. Unknown and unplaced aliases return **404**; inference never discovers
or persists Hub metadata as a side effect.

```bash
curl -X POST http://localhost:52415/v1/images/generations \
  -H 'Content-Type: application/json' \
  -d '{
    "model": "exolabs/FLUX.1-schnell-4bit",
    "prompt": "a fox curled up in autumn leaves",
    "n": 1,
    "size": "1024x1024",
    "response_format": "b64_json"
  }'
```

Request fields:

| Field | Type | Notes |
|-------|------|-------|
| `model` | string | Required placed image model id |
| `prompt` | string | Required generation prompt |
| `n` | integer | Number of images; default `1` |
| `size` | string | `auto` (default), `512x512`, `768x768`, `1024x768`, `768x1024`, `1024x1024`, `1024x1536`, or `1536x1024` |
| `quality` | string | `high`, `medium` (default), or `low` |
| `output_format` | string | `png` (default), `jpeg`, or `webp` |
| `response_format` | string | `b64_json` (default) returns inline base64 image data; `url` stores each image on this API node and returns a fetchable URL instead |
| `stream` | boolean | With `partial_images > 0`, returns an SSE stream of partial and final images instead of one JSON response |
| `partial_images` | integer | Number of intermediate previews per image when streaming; default `0` |
| `advanced_params` | object | Optional `seed`, `num_inference_steps` (1-100), `guidance` (1.0-20.0), `negative_prompt`, and `num_sync_steps` (1-100). When `seed` is omitted, Skulk assigns one so multi-node generation stays deterministic |

The non-streaming response is `{ "created": ..., "data": [...] }` with one
`b64_json` or `url` entry per image.

### Edit images

**POST** `/v1/images/edits`

Image-to-image editing. Unlike generations, this endpoint takes a multipart
form because it carries the input image:

```bash
curl -X POST http://localhost:52415/v1/images/edits \
  -F image=@input.png \
  -F prompt='make it snow' \
  -F model=exolabs/FLUX.1-Kontext-dev-4bit \
  -F response_format=b64_json
```

Form fields mirror the generation fields (`n`, `size`, `quality`,
`output_format`, `response_format`, `stream`, `partial_images`, and a JSON
`advanced_params` string), plus:

| Field | Type | Notes |
|-------|------|-------|
| `image` | file | Required input image, at most 24 MiB raw; larger uploads return **413 Request Entity Too Large** |
| `input_fidelity` | string | `low` (default) or `high`; controls how strongly the input image constrains the edit |

The input image travels to the selected worker over the bounded vision media
path described under chat image inputs; it is never written to the event log
or replicated `State`. The response shape matches image generation.

### Stored images

**GET** `/images`

Lists the images this API node currently stores for `response_format: "url"`
responses. Each entry carries `image_id`, `url`, `content_type`, and
`expires_at`.

**GET** `/images/{image_id}`

Returns one stored image as raw bytes with its stored content type.

```bash
curl http://localhost:52415/images
curl -o out.png http://localhost:52415/images/<image_id>
```

Stored images are node-local and expire one hour after creation; a missing or
expired id returns **404 Image not found or expired**. Use
`response_format: "b64_json"` when you need the image bytes to outlive the
cache.

## Benchmark Endpoints

Benchmark variants of the generation endpoints run the same admission and
validation as their non-bench counterparts, force a non-streaming run
(`stream=false`, and `partial_images=0` for images), and flag the task so the
serving runner collects generation statistics. The response extends the normal
response shape with two extra fields:

- `generation_stats`: runner-reported timing/throughput statistics for the run,
  including the non-identifying `serving_batches` batching-mode flag and
  `in_flight_at_admission` request count. Serving node ids and backend tags are
  redacted from client responses.
- `power_usage`: per-node and total system power sampled from live cluster
  telemetry while the request ran

Endpoints:

- **POST** `/bench/chat/completions` takes the same body as
  `POST /v1/chat/completions` and returns the chat completion plus stats. The
  same `TextGeneration` admission applies.
- **POST** `/bench/images/generations` takes the same body as
  `POST /v1/images/generations`.
- **POST** `/bench/images/edits` takes the same multipart form as
  `POST /v1/images/edits`.

```bash
curl -X POST http://localhost:52415/bench/chat/completions \
  -H 'Content-Type: application/json' \
  -d '{
    "model": "mlx-community/Llama-3.2-1B-Instruct-4bit",
    "messages": [{"role": "user", "content": "Benchmark me"}]
  }'
```

## Cancel an In-Flight Command

**POST** `/v1/cancel/{command_id}`

Requests cancellation of one in-flight generation command by its command ID.
It covers text generation, image generation, embeddings, and speech synthesis
or transcription commands owned by the API node you call: Skulk closes the
local response stream and sends a task cancellation so the serving runner
stops instead of generating into the void.

Finding the command ID:

- streaming chat completions open with the SSE comment line
  `: command_id <id>`, and every streamed chunk's `id` field carries the same
  value
- non-streaming chat responses use the command ID as the response `id`
- Skulk control responses such as `POST /place_instance` return an explicit
  `command_id` field plus the exact resulting `instance_id` (those placement
  commands complete immediately and are not cancellable here)

```bash
curl -X POST http://localhost:52415/v1/cancel/<command_id>
```

A cancelled command returns
`{"message": "Command cancelled.", "command_id": "..."}`. An unknown or
already-completed command returns **404 Command not found or already
completed**. Command streams are node-local, so call the same API node that
accepted the original request. Simply disconnecting from a streaming response
triggers the same cancellation path implicitly.

## Model Discovery

### List models

**GET** `/v1/models`

```bash
curl http://localhost:52415/v1/models
```

This returns known model cards, not just running instances. `GET /models`
serves the same catalog through the same handler; prefer the `/v1/models` path
for OpenAI-compatible clients. Complete installed cards load first so an
air-gapped node keeps the exact generation it can actually launch. Temporary
`qualification_only` installed records are the exception: once a normal signed
catalog card owns the same alias, `/v1/models` reports the signed card and does
not use the retained qualification record to claim that signed generation is
installed. The current supported catalog comes from the external TUF-signed
registry and refreshes at most every 60 seconds; a previously verified catalog
may be used for up to 30 days during an outage. That age limit does not apply to
complete installed artifacts. Bundled cards fill non-installed catalog entries
only when registry access and its acceptable cache are unavailable or disabled.
Registry entries include immutable card and snapshot identities; local custom
cards retain final override precedence.

Each entry also separates discovery truth from runtime truth:

- `registry_architecture` is the trusted open architecture identifier retained
  from the signed catalog.
- `capability_claims` reports signed model/artifact capabilities even when no
  current Skulk engine can serve them.
- `engine_support` reports the active signed engine/build decisions matching
  that exact architecture, artifact format, quantization, capability, and any
  required immutable card identity. These may include experimental or
  unsupported history; placement expands only from exact `supported` claims
  whose build and hardware constraints match a node. Load and feature
  qualification always names the exact card tested, and known-incomplete
  artifact capability evidence blocks expansion.

Installed sidecars retain the intrinsic architecture and capability claims for
air-gapped use. A hash-bound support matrix that was previously TUF-verified is
also usable offline; it never converts an experimental or negative decision
into placement permission.

### Legacy repository-code approvals

**GET** `/models/remote-code-approvals`

Deprecated compatibility endpoint. It lists approval identities retained from
older deployments, but those values no longer participate in model execution.
Signed publication, explicit model addition, and bundled distribution are the
active repository-code authorization boundaries.

Ordinary reads and launch endpoints never fetch an unknown Hugging Face
repository or persist a card implicitly. An external repository must first
enter through authenticated `POST /models/add` or `POST /models/add-card`.

**POST** `/models/remote-code-approvals/{card_id}`

Deprecated compatibility endpoint. Current cards return HTTP 409 because
publication or addition already authorized them; Skulk does not create a
redundant approval entry.

**DELETE** `/models/remote-code-approvals/{card_id}`

Removes an inert approval retained from an older deployment. It does not revoke
a published, bundled, or explicitly added card.

### Search Hugging Face

**GET** `/models/search?query=...&limit=...&mlx_only=...&offset=...`

```bash
curl "http://localhost:52415/models/search?query=qwen3&limit=5"
```

Behavior note:

- `mlx_only=true` restricts results to the `mlx-community` author; the default
  searches all Hugging Face model repositories.
- An empty `query` returns repositories sorted by Hugging Face's trending
  score; text queries sort by downloads.
- `offset` skips that many leading results, for "show more" paging.
- `pipeline_tag` restricts results to one Hugging Face task (for example
  `text-generation` or `automatic-speech-recognition`).
- Ordinary text queries use Hugging Face repository search.
- A query ending in `.gguf` also performs a bounded filename-aware fallback:
  Skulk broadens the model-name prefix, inspects those candidate repositories'
  manifests, and returns only exact filename matches. Exact matches carry a
  `matched_file` repo-relative path so the dashboard can preserve that quant.
- Each result additionally carries discovery metadata when Hugging Face
  reports it: `pipeline_tag` (task), `library_name`, `gated` (license
  acceptance plus an HF token are required to download), `license`,
  `param_count` (total parameters from safetensors or GGUF metadata),
  `total_file_size` (exact GGUF artifact bytes), `context_length`,
  `base_model_repo` and `base_model_relation` (derivation lineage:
  finetune, quantized, merge, or adapter), `arxiv_ids`, `languages`, and
  `architecture`.

### List a GGUF repository's quantizations

**GET** `/models/gguf-quants?model_id=owner/name`

```bash
curl "http://localhost:52415/models/gguf-quants?model_id=unsloth/DeepSeek-V4-Flash-0731-GGUF"
```

Returns `{ "model_id": ..., "options": [...] }` where each option is one
downloadable quantization: its repo-relative first shard (`gguf_file`, the
value to pin when adding or downloading), human `label`, exact `total_bytes`,
and `shard_count`, sorted smallest first. Companion artifacts (speculative
drafters, imatrix calibration files, multimodal projectors) are excluded;
`options` is empty for a non-GGUF repository. The dashboard's Hugging Face
results use this for the per-quant download chooser.

### Fetch a model card summary

**GET** `/models/card-summary?model_id=owner/name`

```bash
curl "http://localhost:52415/models/card-summary?model_id=LiquidAI/LFM2.5-2.6B"
```

Downloads the repository's model card README and returns
`{ "model_id": ..., "summary": ... }`, where `summary` is the card's first
prose paragraphs with markup stripped (bounded length). The summary is empty
when the README is missing or has no usable prose. Summaries are cached per
API process; the dashboard's discovery popovers fetch this lazily when
opened.

### Add a Hugging Face model

**POST** `/models/add`

```json
{
  "model_id": "satgeze/Hy3-1M-GGUF",
  "gguf_file": "hy3-1M-MTP-IQ3_XXS.gguf",
  "source_revision": "0123456789abcdef0123456789abcdef01234567"
}
```

Resolves the repository to one immutable commit, fetches metadata, and adds a
custom model card to the cluster catalog. The
dashboard and operator API may call this through the normal cluster control
surface when the request comes from a direct loopback or trusted-fabric socket
peer (private LAN or CGNAT, with no proxy-forwarding headers and, for browser
requests, an Origin on the same trust classes or naming one of this node's
own hostnames, which is how a dashboard opened via `kite3.local` or the node's
MagicDNS name qualifies while a DNS-rebound attacker hostname does not), or
has passed the authenticated
operator gateway with `operations:write`. The trusted-fabric admission matches
the cluster's standing posture (a peer on those networks can already join the
mesh as a full member) and is what lets a dashboard browsed from another
machine on the LAN add models. Forwarded requests and public peers are still
refused. The explicit add action authorizes
repository code selected by that card; there is no second approval step. A
successful response waits for the exact add command to be ordered, persisted,
and visible in the responding API's catalog, so an immediate download or
placement cannot race catalog convergence. Historical executable custom cards
that lack an immutable source revision fail closed after upgrade; re-adding the
model resolves and persists a pinned revision. A generated GGUF card is
compatible with both llama.cpp engines and prefers
the served `llama_server` tags, so on a node running llama-server it gets
that engine's concurrency slots and is eligible for multi-node pooling via
RPC; nodes without a served binary fall through to the in-process engine.
When the repository exactly matches a bundled card, generated metadata retains
the bundled card's hard pipeline-split constraint; adding a curated model through
the dashboard therefore cannot erase an architecture safety boundary. A
hand-authored card remains an explicit operator override.
The
`model_id` field is required. `gguf_file` is optional; when supplied it must be
an exact repo-relative GGUF weight path and the card pins that quant instead of
using Skulk's default GGUF preference. If the selected quant is split, Skulk
stores its first shard as the backend entrypoint while downloading the full
shard group. `source_revision` is also optional; when supplied it must be a full
40-character Hugging Face commit hash. When omitted, Skulk resolves `main` once
to the Hub's full commit. Card metadata and subsequent artifact downloads are
therefore always pinned to an immutable revision.
When the Hub refuses the metadata fetch with 401 or 403 (a gated or private
repository), the 400 response explains the concrete fix for this node:
configure a Hugging Face token, accept the model terms on the repository page,
or accept them under the same account the configured token belongs to. Like
every HTTPException from this API, the explanation is serialized in the
OpenAI-style error envelope, so clients read it from `error.message`.

### Add an exact unsigned model card

**POST** `/models/add-card`

```json
{
  "model_card": {
    "modelId": "org/model@q4-k-m",
    "sourceRepository": "org/model",
    "sourceRevision": "0123456789abcdef0123456789abcdef01234567",
    "storageSize": {"inBytes": 1234},
    "nLayers": 32,
    "hiddenSize": 4096,
    "supportsTensor": false,
    "tasks": ["TextGeneration"],
    "ggufFile": "model-Q4_K_M.gguf"
  }
}
```

Persists a complete operator-supplied card without fetching or regenerating
Hub metadata. This is intended for exact pre-publication qualification and
other trusted operator workflows. The endpoint preserves the pinned artifact
contract but always forces unsigned custom-card semantics: `is_custom` becomes
true, and every supplied `registry_*` identity, provenance, architecture,
format, or capability claim is removed. Service-bearer installs additionally
receive `qualification_only`; operator installs do not. A full 40-character
immutable `sourceRevision` is required.
Every external vision, MTP, assistant, or draft repository must also declare
its matching immutable companion revision.
The exact add action authorizes repository code selected by the temporary card.
Unlike `/models/add` (which also admits direct trusted-fabric peers for the
dashboard), this mutation accepts only direct loopback access
or an authenticated operator gateway with `operations:write`. A headless
registry qualification worker may instead present
`Authorization: Bearer <token>` when the node configures the same high-entropy
value in `SKULK_EXACT_CARD_QUALIFICATION_TOKEN`. That credential is deliberately
valid only for this exact-card install and
`DELETE /models/custom/{model_id}` cleanup; it grants no general model,
inference, configuration, or operator authority. The service bearer cannot
replace or delete any pre-existing non-qualification card; cleanup requires the
server-assigned `qualification_only` marker. Service cleanup must also send the
complete original candidate as the DELETE body:

```json
{
  "model_card": {
    "modelId": "org/model@q4-k-m",
    "sourceRepository": "org/model",
    "sourceRevision": "0123456789abcdef0123456789abcdef01234567",
    "storageSize": {"inBytes": 1234},
    "nLayers": 32,
    "hiddenSize": 4096,
    "supportsTensor": false,
    "tasks": ["TextGeneration"],
    "ggufFile": "model-Q4_K_M.gguf"
  }
}
```

Skulk applies the same unsigned normalization and deletes only if that complete
temporary card still owns the alias. These ownership preconditions are rechecked
by the elected master when it orders the mutation, so a concurrent operator or
newer qualification change cannot be overwritten or deleted through stale
API-node state.
The master also folds newly refreshed signed-registry truth into this ownership
view, so publication prevents a later temporary override even though registry
refreshes do not traverse the command log.
The endpoint returns success only after the exact card has round-tripped through
that ordering boundary, its originating command ID has been acknowledged after
local persistence/cache application, and the card is visible in the responding
node's catalog. Service-authenticated cleanup likewise waits for its exact
delete event and suppresses retained `qualification_only` installed sidecars
from catalog projection while preserving the downloaded artifact bytes; a
conflicting winner returns `409`, and convergence timeout returns `504`.

### Per-node storage breakdown

**GET** `/store/storage`

Returns the local node's storage picture: every installed artifact across the
configured staging cache, `SKULK_MODELS_DIR`, and read-only model search roots,
with its size, last-use time, and whether a live instance (or one of its companion repos:
MTP sidecar, assistant, vision weights) currently depends on it, plus
event-log usage and free disk on the models volume. Cluster-wide views query
each node's API.

Each artifact entry also reports `installedIdentity`, `manifestSha256`,
`verificationState`, `manifestComplete`, `artifactRole`, and `ownerModelId`.
`locationKind` is `store_local` when the directory belongs to the canonical
store on this node and `node_cache` for a launchable node-local copy.
`registryCardId` identifies the full card retained with the bytes, while a
companion additionally reports its immutable `ownerCardId`; reconciliation
uses those fields to select one current generation per artifact alias.
Directories that cannot be associated with trusted card truth appear with an
`unresolved` verification state and are not imported or launched automatically.

```bash
curl http://localhost:52415/store/storage
```

Staged copies are managed automatically when the model store is on: when an
instance shuts down (and at node startup, which reconciles copies orphaned
by a crash), not-in-use staged models are kept newest-first up to the
`staging_keep_recent_gb` grace budget (default 40 GiB) and evicted beyond
it. Set `cleanup_on_deactivate: false` in the staging config to keep every
staged copy while disk is healthy. Independently, before each store-backed
transfer the worker may evict idle copies inside that grace budget until the
exact additional registered artifact bytes fit with 10 GiB of
operating-system headroom. Base and companion transfers are serialized,
resumable manifest data is credited, and same-filesystem hardlinks count as
zero allocation. Live runners, active model transactions, and the incoming
partial model stay protected; an unsatisfied capacity target becomes
`DownloadFailed` before transfer.

The canonical store host applies the same exact-byte, serialized admission to
its Hugging Face download transaction. It never evicts authoritative models;
insufficient canonical capacity fails the store download before file transfer.
Store-unreachable direct fallback applies the reserve to the node's actual
model-cache filesystem and does not run staging eviction.

## Placement and Instance Management

These endpoints are the heart of the Skulk control plane.

### Quick launch

**POST** `/place_instance`

```bash
curl -X POST http://localhost:52415/place_instance \
  -H 'Content-Type: application/json' \
  -d '{
    "model_id": "mlx-community/Qwen3.5-9B-4bit",
    "sharding": "Pipeline",
    "instance_meta": "MlxRing",
    "min_nodes": 1,
    "excluded_nodes": []
  }'
```

| Field | Meaning |
|-------|---------|
| `model_id` | Exact alias already present in the signed, bundled, installed, or operator-added catalog |
| `sharding` | `Pipeline` or `Tensor` |
| `instance_meta` | `MlxRing`, `MlxJaccl`, or `LlamaRpc` (multi-node GGUF pooling: one driver node holds the model and each donor node lends GPU memory over the network) |
| `min_nodes` | Minimum nodes required for the placement |
| `excluded_nodes` | Optional. Node IDs the master should treat as if absent when scoring this placement. Already-running instances on those nodes are unaffected (exclusion is per-placement, not cluster-wide), and automatic repair re-placements of this instance (memory refusal, download failure) keep honoring the same exclusions. Default: `[]`. Note: node IDs are per-session, so they change when a cluster session restarts. |

The placement is validated against the current cluster state **before** the
command is forwarded, so an impossible placement fails at the API instead of
silently failing on the master:

- **404** when `model_id` is not already present in the authorized local
  catalog. Use the authenticated model-add flow first; launch never performs
  implicit Hub discovery.
- **400** with the specific reason: no connected cycle of `min_nodes` nodes,
  exclusions removed every candidate, every candidate has a positively known
  isolated Zenoh inference data plane, the model does not support Tensor
  sharding, or a node cannot fit its weight shard plus runtime headroom (the
  error names the node and the GB arithmetic).
- **503** when cluster info is still being gossiped (a cluster that just
  formed): connection edges lag node identities by a few gossip rounds, and
  per-node memory info lags the edges. The request internally waits up to
  15 seconds for the info to arrive before giving up, so retry shortly on 503.

The request expresses intent, not a reservation of a prior preview. A card may
declare any number of open backend tags plus an ordered preference. The planner
first removes candidates blocked by participation policy, engine/build
evidence, data-plane health, topology, or capacity; it then ranks
what remains. If preference 1 is unavailable it automatically falls through to
2, then 3, without requiring the client to select a concrete engine or host.
An explicit `excluded_nodes` value remains an operator constraint, not the
normal selection mechanism.

Placement failures preserve the human-readable `error.message` body and include
a stable `X-Skulk-Placement-Failure` response header. Current categories are
`no_valid_placement` and `placement_info_pending`; exact-placement requests may
additionally return `model_card_identity_mismatch`.
`model_code_approval_required` remains in the response schema only for
compatibility with older nodes and is not emitted by current authorization
policy.

`POST /instance` additionally requires every caller-embedded shard card to
match the current effective catalog card exactly, not merely reuse its alias.
This prevents an otherwise valid signed or operator-added identity from being
attached to caller-selected repository code or artifact fields.

A successful response includes both `command_id` and `instance_id`. For
`POST /place_instance` they contain the same stable value: the accepted command
owns exactly that resulting placement identity. Clients should retain
`instance_id` and correlate progress against that runtime rather than guessing
from model name or observation order. The field is additive for older clients.

Memory fitting is checked **per node, not summed across the cycle**: Tensor
sharding splits weights evenly, Pipeline allocates layers proportionally to
each node's available memory, and every node must hold its share times a
runtime-overhead factor (KV cache, activations, runner) on top of the raw
weight bytes. A model that exactly equals a node's free memory is rejected,
because that placement would thrash, not run.

The dry-run uses the same hardware-memory classification as the master. In
particular, a unified-memory GPU may use its combined VRAM/GTT pool for model
fit while fixed-window llama.cpp engines keep their conservative startup
context. The accepted command therefore cannot change context policy between
API validation and master placement.

### Preview valid placements

**GET** `/instance/previews?model_id=...`

```bash
curl "http://localhost:52415/instance/previews?model_id=mlx-community/Qwen3.5-9B-4bit"
```

This is usually the best first Skulk-specific endpoint to call. It shows which
combinations of sharding mode, networking mode, and node count are valid, and
why invalid combinations fail. Each unavailable entry also carries an
`error_code` with the same stable category vocabulary as placement responses.

Each preview's `instance_meta` reports the shape placement would *actually*
mint for that combination, not the shape that was asked about: for example a
GGUF model previewed at two GPU nodes reports `LlamaRpc` (driver plus memory
donors) even though the request enumerates the generic metas. Trust the
preview's reported meta when constructing a follow-up `POST /place_instance`.
The embedded instance's `contextTokenLimit` is also the exact limit launch
would stamp. Unified-memory GPUs are not previewed with a discrete-VRAM context
lift that the master would later remove.

Besides the planner's ranked pick per shape, the response also contains
per-host single-node previews marked `"alternative": true` for every other
host that passes admission. On a heterogeneous fleet the ranked winner is
typically the node with the most free accelerator memory; the alternatives
explain the full set of valid hosts. They are not reservations: ordinary
`POST /place_instance` recomputes and launches the best candidate against
current facts. Operators who intentionally want to steer policy may use node
exclusions and request a fresh preview. Alternatives are omitted when
`node_ids` already constrains the hosts.

Every preview additionally reports `compatibility_source` (`card` or
`signed_engine_support`), the exact `support_claim_ids` used when a signed
matrix expanded placement, and an operator-readable `compatibility_detail` on
engine/build, hardware, or incomplete-artifact gaps. A positive matrix claim
must match the node's advertised exact build and hardware class. Missing,
stale, experimental, or unsupported claims do not widen placement; legacy
`compatible_backends` remain valid for existing cards.

| Query parameter | Meaning |
|-----------------|---------|
| `model_id` | Required. Hugging Face-style model ID. |
| `node_ids` | Optional, repeatable. Restricts previews to candidate cycles that contain *all* of these node IDs (subset matching). |
| `excluded_node_ids` | Optional, repeatable. Excludes the listed node IDs from candidate cycles for every previewed combination. Mirrors the `excluded_nodes` field on `POST /place_instance` so dashboards can render an accurate preview against the post-exclusion topology. |

```bash
# Preview with one node excluded:
curl "http://localhost:52415/instance/previews?model_id=mlx-community/Qwen3.5-9B-4bit&excluded_node_ids=12D3KooWAbc..."
```

### Build a placement manually

**GET** `/instance/placement`

Use this when you want a specific combination and want to inspect the exact
instance shape before launch, including the hardware-aware
`contextTokenLimit` that the master would stamp.

### Create an instance from a fully specified placement

**POST** `/instance`

Use this when you already have an `instance` object and want exact control. A
successful response returns the accepted `command_id`, the submitted
`instance_id`, and its `model_card`; clients can use that exact instance
identity to correlate the acknowledgement with later runtime and failure truth.
The API requires every embedded shard card's `modelId` to match the assignment's
canonical `modelId` before acknowledging creation. An inconsistent shard card
returns HTTP 400 with `X-Skulk-Placement-Failure:
model_card_identity_mismatch`, and no instance state is created.

Persist the submitted instance identity before sending. HTTP acceptance is not
download or runner readiness. If the response is lost, reconcile that exact ID
against `GET /state` and `instanceFailures`; this endpoint does not document a
client idempotency-key guarantee. See [Controller integration](controller-integration.md)
for bounded readiness observation and cleanup responsibilities.

### Inspect one instance

**GET** `/instance/{instance_id}`

### Delete an instance

**DELETE** `/instance/{instance_id}`

## Intelligent Fabric

When intelligent-fabric mode is enabled in the cluster configuration
(`intelligent_fabric.enabled`), the fabric keeps a small resident model (the
steward) placed as a hidden system instance. The steward investigates the
cluster through a bounded tool surface and answers operator questions. Its
read tools return evidence; its basic-action tools can only create inert,
expiring proposals. Read-only questions remain available to ordinary clients,
but the server exposes proposal-creation tools to the model only when the chat
request has trusted-fabric or authenticated operator-gateway mutation authority.
The model never receives a direct mutating tool, and a separately authenticated
operator must approve the exact proposal before the master can dispatch it.
`steward` remains the internal role and compatibility
identifier, but operator surfaces present this cognition as Skulk itself rather
than as a separate assistant or character.

### Talking to Skulk: the virtual model

Clients talk to Skulk through the standard OpenAI-compatible
`POST /v1/chat/completions` endpoint using the reserved model id
`skulk/steward`. Any OpenAI-compatible client works, streaming included; no
steward-specific client code is required beyond the model id.

Semantics of the reserved id:

- The server runs the steward's investigation loop (up to 8 tool calls per
  turn: cluster state, node resources, telemetry and data-plane
  diagnostics, version status, performance envelopes, named-node doctor
  results, the model catalog, and a search over Skulk's own bundled
  documentation, plus inert basic-action proposal tools) and answers from the
  evidence. `get_node_diagnostics`
  requires a friendly `node_name` and returns that node's complete diagnostic
  bundle; `run_doctor` also requires `node_name` and returns the selected
  node's bounded doctor findings. Both resolve only unique live friendly names
  and refuse missing or ambiguous targets rather than exposing node IDs.
- Ordinary clients receive the same read tools and may ask diagnostic or
  advisory questions. The four proposal tools are included only for a direct
  trusted-fabric request or an authenticated operator-gateway request; this
  prevents public chat access from filling the bounded proposal queue.
- The tool trace is returned as reasoning content: in streaming responses,
  each tool step arrives as a `reasoning_content` delta while the
  investigation runs, followed by the answer as `content`; non-streaming
  responses carry the trace in the message's `reasoning_content` field.
- Client-supplied `tools` are rejected with `400`: the steward's tool
  surface belongs to the server.
- Client `system` messages are ignored in favor of the steward's own system
  prompt; `user` and `assistant` turns form the conversation history, which
  the client owns and resends each turn (the server is stateless).
- The steward always runs with thinking disabled, regardless of what the
  brain model supports. Requests to the reserved id cannot turn it back on;
  addressing the underlying card directly still gives you the model's normal
  reasoning behavior.
- Requests to the id while intelligent-fabric mode is disabled return `404`
  with an explanatory message.
- If the mode is enabled but the steward is not ready to answer (still being
  placed, still downloading its weights, still loading), the request is
  refused with `503` before any streaming begins, carrying a `Retry-After`
  header and a JSON body whose `detail` is the `GET /v1/steward` payload
  (`enabled`, `present`, `ready`, `steward_model`, `instance_id`, `state`)
  plus a human-readable `message`. Clients should back off and retry, or
  poll `GET /v1/steward` and show the `state` while they wait.
- A steward that disappears in the window between that check and dispatch
  (a repair starting at exactly the wrong moment) still surfaces as an error
  chunk in the normal chat-completions error shape, because the response has
  already begun by then.
- The underlying model card id (for example the bundled Qwen3.6-35B-A3B)
  remains addressable as an ordinary model and answers WITHOUT tools or
  cluster access: only the reserved id selects model-plus-harness.

The steward appears in `GET /v1/models` as an entry flagged with
`system_role: "steward"` while the mode is enabled, so model pickers can
recognize the fabric cognition without listing it as an ordinary model. Its
operator-facing `name` is `Skulk`; clients should retain the `system_role`
value only for discovery and compatibility.

#### Extensions on a steward turn

If the serving node has an extension installed that provides chat middleware,
its two hooks run on steward turns as well as on ordinary chat completions. A
node with no such extension behaves exactly as described above.

The turn is presented to middleware in the same canonical form the ordinary
chat path uses: the steward's system prompt as `instructions`, and the `user`
and `assistant` history as `input`. Only those two channels are read back:

- **`instructions`** becomes the turn's system message, so a middleware can
  augment (or replace) the steward's prompt. This is how, for example, an
  ambient-memory extension adds recollections from earlier conversations.
- **`input`** becomes the conversation history, keeping only `user` and
  `assistant` messages with non-empty content.
- Everything else a middleware returns is ignored. The model, sampling
  parameters, and tool surface belong to the steward, so a middleware cannot
  reroute the turn to another model or arm a different tool set.

If a transform leaves the turn without a trailing `user` message, the whole
transform is discarded (prompt and history both) and the turn runs on the
operator's original conversation, because a steward turn exists to answer an
operator question. Whatever a middleware returns, the params handed to the
response observer always describe the turn that actually ran: the reserved
model, the filtered history, and the effective system prompt.

The response observer fires exactly once per steward turn, receiving the final
answer. The investigation's individual tool steps and the steward's periodic
liveness probe are not observed: they are internal machinery, not
conversations. A middleware that raises is logged and skipped, and the steward
answers as though no extension were installed. None of this changes the
request or response wire format.

### GET /v1/steward

Returns steward availability. Clients use this to decide whether to show a
steward surface at all.

Response fields:

- `enabled`: whether intelligent-fabric mode is enabled in Settings.
- `present`: whether a steward placement currently exists.
- `ready`: whether every steward runner reports Ready or Running. Present
  but not ready means the model is still downloading or loading; clients
  should keep showing a preparing state and hold chat until ready.
- `steward_model`: model card id of the steward brain when present, else null.
- `instance_id`: the steward instance id when present, else null.
- `desired_model`: the better brain currently being prepared, or the serving
  brain when no transition is active.
- `transition`: controlled brain lifecycle: `idle`, `prestaging`, `replacing`,
  or `repairing` after the placement disappears.
- `progress`: aggregate prestaging completion from `0` to `1` when byte totals
  are available, else null.
- `state`: a one-word lifecycle summary derived from the fields above plus
  the liveness canary's history, for clients that want to render a single
  line instead of re-deriving the precedence rules. The booleans remain
  authoritative. Values:
  - `disabled`: intelligent-fabric mode is off.
  - `downloading`: a placement exists and the brain's weights are still
    being staged. This is the long first-run wait.
  - `starting`: the fabric is placing the steward, or it is placed and
    loading.
  - `ready`: serving, with no outstanding liveness failure.
  - `degraded`: serving, but the elected API node's liveness canary has at least
    one failed probe outstanding. The steward may still answer; three
    consecutive failures make the fabric replace the placement.

Note: deleting the steward instance through `DELETE /instance/{instance_id}`
is refused with `409` while intelligent-fabric mode is enabled; disable the
mode in Settings to remove the placement (the fabric then tears it down
automatically).

### GET /v1/steward/proposals

Returns the bounded, newest-first audit of steward action proposals. This route
uses the same trusted-fabric or authenticated operator-gateway authorization as
operator mutations. It does not expose internal node IDs, instance IDs, command
IDs, or embedded model-card payloads.

Each response item contains:

- `proposal_id`, `created_at`, and `expires_at`;
- `action`: `place_model`, `stop_model`, `restart_model`, or `cancel_download`;
- a safe `target` label, `rationale`, bounded `evidence`, and `expected_effect`;
- `status`: `pending`, `approved`, `dispatched`, `rejected`, `expired`, or
  `failed`; and
- optional `decided_at`, safe actor class in `decided_by`, and `outcome`.

Proposals created by the built-in harness expire after ten minutes. The master
refuses already-expired proposals, proposals with a lifetime over fifteen
minutes, automatically publishes expiry when the deadline passes, and refuses
more than 32 simultaneously pending proposals. State targets a 128-record audit
window by pruning the oldest completed terminal records; pending, approved, and
dispatched records still inside their five-minute failover-recovery window are
never pruned to make room.

The action set is intentionally narrow:

- `place_model` revalidates exact authorized catalog truth and runs the normal
  placement planner. Normal placement also stages missing shards.
- `stop_model` deletes the exact ordinary instance selected when proposed.
- `restart_model` first tears down that exact ordinary instance, records
  `approved`, then re-places the captured intent only after replicated deletion
  and live capacity truth show that the old allocation has been released. It
  fails if replacement capacity does not become available within five minutes.
- `cancel_download` cancels the exact model transfer attempt on the selected
  named node. Approval fails closed if that attempt finishes or is replaced by
  a retry, and the worker repeats the attempt-identity check when the command
  arrives so a later retry cannot be cancelled.

System-role instances are never eligible. Targets are resolved before proposal
creation and revalidated by the elected master at approval, so stale or
ambiguous proposals fail without mutation.

### POST `/v1/steward/proposals/{proposal_id}/decision`

Submits one explicit operator decision for a pending proposal. The path
parameter is the `proposal_id`; the JSON body is:

```json
{"approved": true}
```

`approved: false` rejects the proposal. This route requires trusted-fabric or
authenticated operator-gateway authority. Decisions are single use. Missing
proposals return `404`; proposals already decided in the local replicated view
return `409`.

A successful response contains the proposal ID, the decision command ID, and an
acceptance message. Acceptance means the decision entered the ordered command
path. Clients should poll `GET /v1/steward/proposals` for the master-authoritative
result. `dispatched` means the approved action was translated into and accepted
by an existing typed placement, deletion, replacement, or download command; it
does not claim that an asynchronous model start, stop, or download has already
completed. For restart only, `approved` means the decision is durable and the
planning loop will dispatch teardown before waiting for released capacity.
Stop and restart download cleanup likewise waits for the replicated decision
before it is forwarded. Restart also revalidates its captured model-card
identity before teardown. For five minutes after the separate timestamp of any
`dispatched` transition, a promoted master compares the proposal's exact command
identity with replicated state and reissues a missing action effect once.
Download cancellation uses an additional durable step: the replicated
`approved` decision is indexed before dispatch is armed. The armed `dispatched`
transition must then be indexed before a later planning pass forwards the
attempt-bound cancellation.

Setting `SKULK_FABRIC_CAPABILITIES_DISABLE=1` on the elected master is the global
fail-closed kill switch. It converts an otherwise valid approval into `failed`
without dispatching the proposed action, and a promoted master fails carried
dispatch recovery rather than reissuing its effect. There are no autonomous approvals or
per-action grants in this release: every proposal requires a separate operator
decision.

## Download Management

### Start a node download

**POST** `/download/start`

Lower-level endpoint for explicit node download control. It accepts a target
node and shard metadata only from direct loopback or trusted-fabric callers
(private LAN or CGNAT socket peers without proxy-forwarding headers; browser
requests must also present an `Origin` on those trust classes or naming one of
this node's own hostnames) or an authenticated operator gateway. The embedded model card must exactly match current authorized
catalog truth apart from a snapshot-only publication stamp; unknown aliases
return `404`, and stale or forged content returns `409` without dispatching a
download.

### Delete a node download

**DELETE** `/download/{node_id}/{model_id}`

## Model Store Endpoints

These endpoints are available when the model store is configured.

If it is not configured, Skulk returns `503 Store not configured`.

### Store health

**GET** `/store/health`

Use this to confirm whether the store is configured and reachable.

### Store registry

**GET** `/store/registry`

Use this to inspect which models the shared store knows about.

Each registry entry records nullable `source_revision` and `source_repository`
metadata. Cache hits require the effective repository and revision to match, so
an unchanged alias cannot reuse bytes from a different signed source.

For registry v2 cards, the nested installed card may also contain
`artifact_bundle`: its exact repository-relative root and required-file
manifest, immutable bundle identity, download size, and equivalent alternate
locations. Bundle identity is part of installed-generation matching, allowing
multiple card aliases from one repository and revision without a store-key
collision. V1 cards omit this additive field and retain prior behavior.

Entries also include the full `installed_card` record, verification state,
artifact role and owning card, `current_registry_identity`,
`installed_not_current`, `update_available`, active signed `advisories`,
`cached_on_nodes` (identity, completeness, bytes, last use, in-use state, and
`location_kind`), and reconciliation state plus last verification time.
`location_kind` distinguishes canonical `store_local` availability from a
`node_cache` copy. Companion artifacts are first-class entries grouped under
their owning base card by the dashboard.

The top-level `cache_inventory` reports `observed_nodes`, `expected_nodes`, and a
coverage state. Its additive `store_nodes` list identifies live nodes currently
advertising the canonical-store role, including when a legacy entry has not yet
resolved to an exact installed identity:

- `syncing`: at least one newly live node has not published its first inventory.
- `current`: every live node has a fresh, complete reading.
- `degraded`: known locations are partial because a reading is stale, missing
  after convergence, or truncated by the fixed telemetry bound.
- `unavailable`: no usable node inventory exists.

Canonical store entries are not copied into telemetry. Store hosts publish only
their role. Each API synthesizes exact `store_local` entries when an installed
identity is available; clients can combine `store_nodes` with canonical store
truth for unresolved legacy entries without weakening identity verification.
Other cache locations come from compact, last-write-wins node telemetry. Stale
known locations remain visible while the top-level state is `degraded`; callers
must not treat this operator/read projection as transfer authorization.
Reconciliation continues to query `GET /store/storage` directly and verifies
the exact installed identity and manifest before import or export.

The response is `{"entries": [...], "cache_inventory": {...}}` and is fully described as
`StoreRegistryResponse` in generated OpenAPI.

The dashboard combines registry results with `GET /v1/models` metadata so it can
display derived tags such as `vision`, `thinking`, `embedding`, `tensor`, and
`optiq` in the Store list.

**GET** `/registry` (internal model-store transport port)

Returns the authoritative store index consumed by Skulk nodes. The optional
`recover_installed_cards=true` query first rebuilds installed-card associations
from complete local sidecars and trusted catalog cards. Every entry is emitted
with JSON-native values; in particular, the nested installed card's
`captured_at` value is an ISO 8601 UTC string rather than a native datetime.
This endpoint is cluster-internal transport; operators and dashboards should
use the enriched public `GET /store/registry` endpoint above.

### Store downloads

**GET** `/store/downloads`

Use this to inspect shared-store download activity. The listing carries
pending, in-progress, and failed downloads; each failed entry keeps an
actionable `error` explanation (for example how to authenticate for a gated
Hugging Face repository) and stays listed until a retry replaces it or the
store host restarts. Cancelled downloads are not listed.

### Request a store download

**POST** `/store/models/{model_id}/download`

Use this when you want the store host to fetch and register a model.

The response reports the store's current transfer state. A store-host
rejection (for example an immutable-card conflict or a capacity limit) is
reported in the same 200 response with `status` set to `error` and the
store's operator-readable reason in `error`; callers must check the body's
status rather than treating any 200 as an accepted transfer.

For signed-registry artifacts, Skulk's internal request also carries the
immutable card ID. The store host verifies that identity against its own signed
catalog and applies the synchronized cluster repository-code decision before
fetching bytes. A v2 card makes the signed bundle manifest authoritative: only
its required files are fetched, directory layout is preserved, and every
declared size and available upstream object identity is verified. Trust does
not depend on which node initiated the request.

The optional JSON body accepts the following fields:

- `gguf_file`: non-empty repo-relative GGUF path selecting the base or companion
  quant.
- `extra_gguf_files`: list of non-empty repo-relative GGUF paths to co-fetch from
  the same repository, such as a same-repo served draft.
- `source_revision`: full 40-character immutable Hugging Face commit.
- `source_repository`: upstream `owner/repository` containing the bytes when the
  store entry's `model_id` is an alias; identifiers longer than 512 characters
  are rejected.
- `registry_card_id`: immutable `card_<content-derived-id>` selecting the signed
  base-card generation.
- `artifact_bundle_id`: immutable `bundle_<content-derived-id>` selecting the
  exact v2 file bundle. Automated pre-publication qualification uses this pin
  so an alias replacement cannot redirect the download.
- `owner_model_id`: owning base-model alias for a companion artifact. It is
  required for non-`base` roles and must be an `owner/model` identifier no longer
  than 512 characters.
- `owner_registry_card_id`: immutable signed identity of that owning base card.
  Omit it only for bundled or custom owner cards without a registry identity.
- `artifact_role`: one of `base`, `vision_weights`, `mtp_sidecar`, `assistant`,
  `served_draft`, or `vllm_draft`; defaults to `base`.

A complete companion request names the companion repository and immutable
revision, its role, and its owning card:

```json
{
  "gguf_file": "draft.gguf",
  "extra_gguf_files": [],
  "source_revision": "0123456789abcdef0123456789abcdef01234567",
  "source_repository": "owner/draft-repository",
  "owner_model_id": "owner/base-model-alias",
  "owner_registry_card_id": "card_<content-derived-id>",
  "artifact_role": "served_draft"
}
```

`gguf_file` pins which quant the store fetches for a multi-quant GGUF repo (that
file's shard group plus `config.json`). `source_revision` pins all repository
reads to a full immutable Hugging Face commit. When either value is omitted and
a curated model card declares it, the endpoint uses the card value. Otherwise,
omitting `source_revision` follows mutable `main`. A GGUF pin naming a file not
present in the selected revision falls back to the default at the store protocol
layer; the `/models/add` card-building endpoint validates exact pins before
requesting a download.

When `registry_card_id` is omitted, Skulk selects the current card when one
exists. A Hugging Face search result absent from the catalog may still be
downloaded with no card ID; the store records it as unverified rather than
claiming signed registry provenance. Supplying `registry_card_id` requests that
exact immutable generation and returns `409 Conflict` when the store host cannot
verify it.
Supplying `artifact_bundle_id` additionally requires both the API node and the
canonical store to resolve that exact bundle for the alias. A changed, missing,
or legacy bundle returns `409 Conflict` instead of downloading different bytes.
Companion requests instead bind `owner_registry_card_id` to `owner_model_id` and
require the repository, revision, selected file, and `artifact_role` to match
that owning card's signed companion declaration. A mismatched alias, role, or
artifact selection returns `409 Conflict`; malformed identifiers and roles
return `400 Bad Request`.

The response reports `modelId`, nullable `sourceRevision`, `status`, and
`progress`; transport or store rejection additionally reports `error`. The
generated `StoreDownloadResponse` schema describes this wire contract.

### Reconciliation status

**GET** `/store/reconciliation`

Returns `state`, `inventory_only`, scanned node and discovered/imported artifact
counts, pending identities, failures, and `last_verified_at`. Automatic imports
are enabled by default (`inventory_only: false`); inventory-only is an optional
production rollout mode rather than a prerequisite. If cluster configuration
removes or disables the local store or automatic reconciliation, the lifetime
task becomes idle and checks for restored eligibility every ten seconds. It
resumes automatic scans when this node becomes an enabled store host again,
without restarting the API. An already running pass keeps its original interval
before the next eligibility check.

**POST** `/store/reconciliation/rescan`

Runs one immediate retry. This mutation accepts only a loopback socket peer,
rejects proxy forwarding headers, and requires a loopback browser origin when
an `Origin` header is present.

### Internal cache export

**POST** `/store/internal/exports`

Creates a random short-lived capability bound to one installed identity,
manifest digest, target store node, byte ceiling, and expiry. The caller's
socket address must also match an advertised interface of the claimed store
node; the node-id field and header are not accepted as self-asserted identity.

**GET** `/store/internal/exports/{capability_token}/{relative_path}`

Serves only paths in the granted manifest, requires the bound target-node
header, rejects files changed after capability issuance, supports HTTP byte
ranges for restart recovery, and enforces the capability's cumulative byte
ceiling across requests. These endpoints are internal reconciliation transport,
not a public model-download API.

**POST** `/imports` (internal model-store transport port)

Asks the authoritative store to pull and atomically publish one artifact from a
node cache. This endpoint is internal reconciliation transport and accepts only
a direct loopback socket peer with no proxy-forwarding headers; remote or
forwarded callers receive `403`. The JSON body contains:

- `record`: the complete versioned `InstalledCardRecord`, including its full
  card and canonical file manifest.
- `source_base_url`: the selected source node's internal export URL.
- `capability_token`: the short-lived token issued by that source for this
  manifest and target store node.
- `target_node_id`: the store node identity bound into the capability.

The store resumes individual files with HTTP ranges, enforces its capacity
floor, verifies every size and SHA-256 digest, writes the installed-card
sidecar, and publishes the new generation and rebuildable registry entry only
after complete verification. A peer record claiming `registry_verified` is
also rebound to the store host's independently TUF-verified card: its full
immutable card payload, alias, repository, revision, selected file, artifact
role, and companion ownership must agree before any transfer starts. A
successful response is the resulting store registry entry. Malformed records
or missing fields return `400`; invalid or expired capabilities, unsafe
manifest paths, insufficient capacity, source loss, digest mismatch, or signed
card disagreement fail the import without replacing the active generation.
Operators do not call this endpoint directly; the reconciler uses it after
`POST /store/internal/exports` grants a transfer.

### Store download status

**GET** `/store/models/{model_id}/download/status`

### Cancel a store download

**DELETE** `/store/models/{model_id}/download`

Cancels one pending or active canonical-store download. Partial files remain in
the store staging directory so a later request can resume instead of starting
over. Repeating cancellation for an already-cancelled transfer succeeds. The
endpoint returns `409` when no cancellable transfer exists.

### Delete a model from the store

**DELETE** `/store/models/{model_id}`

Removes the model from the store host (registry + disk) and broadcasts a
cluster-wide eviction so every node also drops its locally-staged copy, freeing
disk fleet-wide instead of leaving worker copies until they age out under
staging pressure. Returns `404` if the model is not registered in the store. (To
clear staged copies without deleting the store copy, use
`POST /store/purge-staging`.)

Before deleting bytes, the store durably tombstones the alias against automatic
reconciliation. A stale node cache that missed eviction remains visible in
`cached_on_nodes` but cannot recreate the base artifact or its owned companions.
The tombstone survives restarts and is cleared only after a later explicit store
download for that alias completes successfully.

### Purge staging caches

**POST** `/store/purge-staging`

Use this to remove staged model artifacts from nodes without deleting the store
copy itself. Each artifact directory is removed as one unit, so its model bytes,
installed-card sidecar, revision markers, and last-use marker are evicted
together.

### Start optimization

**POST** `/store/models/{model_id}/optimize`

Use this for workflows such as model optimization or alternate artifact generation.

## Models Endpoint

### List models

**GET** `/v1/models`

Returns the known model catalog, including downloaded models and catalog-backed
entries. Each item includes nullable `source_revision` metadata identifying the
qualified Hugging Face commit when its card pins immutable artifacts.

When an alias has an active installed generation, artifact, capability, runtime,
trust, and `registry_card_id` fields describe that retained full card. The
separate `current_registry_identity` and `update_available` fields describe a
newer signed catalog generation without pretending it is already active.

Important fields:

| Field | Type | Meaning |
|-------|------|---------|
| `id` | string | Canonical model ID |
| `capabilities` | array | Functional capabilities such as `text`, `vision`, `thinking`, `code`, `embedding`, `tts`, or `stt` |
| `tags` | array | UI-friendly derived labels such as `vision`, `thinking`, `embedding`, `tts`, `stt`, `tensor`, and `optiq` |
| `supports_tensor` | boolean | Whether tensor parallel launch is supported |
| `base_model` | string | Base family or upstream source model when known |
| `artifact_repository` | string | Upstream repository containing the artifact bytes; may differ from `id` when several exact files or quants share one repository |
| `artifact_file` | string or null | Exact selected file for file-addressed artifacts such as GGUF |
| `catalog_source` | string | `registry`, `bundled`, or `custom` |
| `registry_card_id` | string or null | Immutable content-derived identity of the active installed card, or the effective catalog card when not installed |
| `registry_snapshot_id` | string or null | Signed catalog snapshot that supplied the card |
| `registry_provenance` | string or null | Audited signed-registry origin (`foxlight`, `agent`, or `community`); null for bundled/custom cards |
| `installed` | boolean | Whether the authoritative cluster store has a complete active generation, falling back to the API node's local sidecar when the store has no record |
| `active_installed_identity` | string or null | Durable identity of that cluster-store generation, or the node-local fallback generation |
| `installed_verification` | string or null | `registry_verified`, `local_legacy`, `custom`, or `unresolved` |
| `current_registry_identity` | string or null | Current signed identity for the alias, which may differ from the active install |
| `update_available` | boolean | A newer signed generation exists but is not active until transfer commits |
| `advisories` | array | Active signed warn-only security notices affecting the installed or current card |
| `remote_code_approval_required` | boolean | Deprecated compatibility field; current cards return `false` because publication, explicit addition, or bundled distribution is the authorization boundary |
| `remote_code_trust_identity` | string or null | Deprecated identity from the retired secondary approval ceremony; current cards return `null` |
| `remote_code_approved_for_cluster` | boolean | Deprecated compatibility state; current cards return `false` |
| `remote_code_approved_on_this_node` | boolean | Deprecated compatibility alias for `remote_code_approved_for_cluster` |
| `remote_code_automatically_trusted` | boolean | Whether repository code is authorized by signed publication, explicit addition, or bundled distribution for this exact card |
| `audio` | object | Declared speech metadata from the model card, including `kind`, audio response formats, streaming/realtime flags, built-in `voices`, `default_voice`, voice/reference-audio flags, translation support, and sample rates |
| `resolved_capabilities.supports_speech_synthesis` | boolean | Whether clients should treat the model as a text-to-speech model |
| `resolved_capabilities.supports_transcription` | boolean | Whether clients should treat the model as a speech-to-text model |
| `resolved_capabilities.supports_speech_translation` | boolean | Whether clients should treat the model as supporting speech translation |
| `resolved_capabilities.supports_audio_output` | boolean | Whether the model produces audio output |
| `resolved_capabilities.supports_realtime_audio` | boolean | Whether the model declares realtime audio support |
| `resolved_capabilities.audio_response_formats` | array | Encoded audio formats the model can produce for speech synthesis |
| `runtime.mtp_sidecar_repo` | string | Repo of this model's MTP sidecar (prediction heads), when it declares one |
| `runtime.mtp_sidecar_revision` | string | Immutable commit of a separately hosted MTP sidecar |
| `runtime.assistant_model_repo` | string | Repo of this model's speculative-decoding assistant (drafter), when it declares one |
| `runtime.assistant_model_revision` | string | Immutable commit of a separately hosted assistant model |
| `runtime.served_spec_draft_repo` | string | Repo of this model's separate served-engine draft GGUF, when it declares one |
| `runtime.served_spec_draft_revision` | string | Immutable commit of a separately hosted served-engine draft |
| `runtime.vllm_spec_draft_repo` | string | Repo of this model's separate vLLM drafter, when it declares one |
| `runtime.vllm_spec_draft_revision` | string | Immutable commit of a separately hosted vLLM drafter |

The dashboard uses `tags` for compact badges and `capabilities` for filtering
and richer tooltips. The `audio` and `resolved_capabilities.*speech*` fields
identify speech-capable models; `supports_speech_synthesis` models can serve
non-streaming `/v1/audio/speech` when mounted, and `supports_transcription`
models can serve non-streaming `/v1/audio/transcriptions`. The chat dashboard
uses the same metadata to show TTS playback and STT microphone controls only
for ready mounted speech models. Browser microphone capture is a browser
security feature, so STT recording controls require a secure origin such as
HTTPS or localhost even though the API endpoint itself is ordinary multipart
HTTP. Speech translation metadata remains reserved for later audio endpoints.
The four `runtime.*_repo` fields name a model's
speculative-decoding companions (a draft model or an MTP-head sidecar). Those
companion repos are downloaded and loaded automatically with their parent and
are not independently placeable, so the dashboard marks any store entry matching
one of these repos as a companion (a "Drafter" or "Sidecar" badge) rather than
offering it launch, placement, or optimize actions.
For signed cards, every separately hosted companion has a matching full commit
revision; companions stored in the base artifact repository inherit the base
card's `source_revision`.

## Configuration Endpoints

### Get config

**GET** `/config`

Returns the current cluster config and config path. Sensitive values (`hf_token`) are stripped from the response.

The response also carries an `effective` block describing runtime-resolved values that are not part of the persisted file:

- `kv_cache_backend`: the KV cache backend actually in effect (config value or `SKULK_KV_CACHE_BACKEND` override)
- `has_hf_token`: whether a HuggingFace token is configured (via the file or `HF_TOKEN`), without exposing the token
- `experimental_mode_enabled`: whether this node runs with `SKULK_ENABLE_EXPERIMENTAL_MODE` set; when a release carries active experiments, the dashboard uses it to reveal the gated Experiments settings section

The persisted `experiments` section is deprecated compatibility surface: every
speech feature that incubated there has graduated to standard, and no built-in
experiment is currently active. The fields remain accepted (the strict config
would otherwise refuse an existing `skulk.yaml` that carries them) but are all
ignored:

- `experiments.tts_streaming`: deprecated compatibility field. Stable TTS
  streaming ignores this value and follows mounted model capability metadata.
- `experiments.stt_realtime`: deprecated compatibility field. It remains
  accepted in existing configuration but is ignored; realtime STT is selected
  from card truth, reachable transport, and ready mounted capacity.
- `experiments.speech_translation`: deprecated compatibility field. Speech
  translation is a standard capability; `/v1/audio/translations` serves for
  any mounted card declaring `audio.supports_translation = true`, and this
  value is accepted but ignored.

The optional `model_trust.approved_remote_code_identities` list is deprecated
compatibility state from the retired secondary repository-code approval
ceremony. Strict config parsing still accepts and preserves the values during
rolling upgrades, but current placement, store, and runner paths do not consult
them. The dashboard no longer presents a Model trust editor.

### Update config

**PUT** `/config`

Updates cluster-wide config. Important behavior:

- if you omit `hf_token`, Skulk preserves the existing value
- if you omit `logging`, Skulk preserves the existing logging config
- if you omit `experiments`, Skulk preserves the existing experiment toggles
- if you omit `model_trust`, Skulk preserves the deprecated compatibility state
  during rolling upgrades
- `model_trust` cannot be replaced through `PUT /config`; authenticated
  operators receive `409` because the current API has no secondary model-trust
  ceremony
- `hf_token` propagates over the PSK-encrypted cluster fabric: a token
  entered in any node's Settings converges onto every node, including the
  model store host that actually fetches from Hugging Face. A broadcast
  carrying no token (or a blank one) never erases a receiving node's existing
  local token; every write remains atomic and owner-only (mode `0o600`), and
  `GET /config` still never returns the token
- logging changes (enable/disable) take effect immediately on all nodes
- inference changes affect future launches
- historical model-trust commands and state remain wire-compatible but have no
  effect on current authorization
- model-store location changes generally require restart

### Filesystem browse

**GET** `/filesystem/browse`

Used by the dashboard to browse a safe subset of the filesystem when selecting config paths.

### Node identity

**GET** `/node/identity`

Returns hostname, preferred IP, and node identity information used by the dashboard.

`GET /node_id` is the minimal companion route: it returns only this node's ID
(the same value `/node/identity` reports, without the hostname and IP fields).
Node IDs are per-session and change when the process restarts.

### Restart a node

**POST** `/admin/restart?node_install_id=<stable id>`

Gracefully restart the Skulk process on this or a remote node. Operator clients
should pass the stable UUIDv4 `node_install_id` published under
`GET /state` → `nodeIdentities[*].nodeInstallId`. Skulk resolves that identity
against current live telemetry immediately before dispatch, so a process restart
cannot leave the client targeting an expired libp2p session. A missing stable
target returns HTTP 404; an ambiguous target returns HTTP 409.

Legacy local clients may continue to pass the session-scoped
`node_id=<runtime id>`. Supplying both target forms returns HTTP 400. Omitting
both restarts the API node itself. A local target replaces the current process
image in-place via `os.execv` (same PID); a remote target sends the existing
`RestartNode` command via pub/sub.

- GPU/Metal memory is released when the process image is replaced
- the node rejoins the cluster automatically on startup
- active inference is interrupted

Returns `{"status": "restarting", "node_id": "...", "node_install_id": "..."}`
for stable local targets, or the corresponding `"restart_sent"` status for
stable remote targets. Legacy session-targeted responses retain their existing
shape without `node_install_id`.
If a local restart is already scheduled, returns HTTP 409 with `{"status": "restart_already_pending"}`.

### Onboarding status

**GET** `/onboarding`

Returns whether the dashboard onboarding flow has been completed on this node:

```json
{"completed": false}
```

**POST** `/onboarding`

Marks the local onboarding flow as complete and returns `{"completed": true}`.
The request takes no body. The flag is a node-local marker file, not cluster
state: each node tracks its own onboarding status, and the dashboard uses it to
decide whether to show the first-run setup flow.

```bash
curl http://localhost:52415/onboarding
curl -X POST http://localhost:52415/onboarding
```

## State, Events, and Tracing

### Cluster state

**GET** `/state`

Returns the cluster state as Skulk currently sees it.

`instanceFailures` is a newest-first, event-sourced history of the 64 most
recent terminal placement failures. Each entry includes the vanished
`instanceId`, `modelId`, optional `systemRole`, stable `errorCode`, bounded
operator-safe `errorMessage`, assigned `affectedNodeIds`, and UTC `recordedAt`.
Ordinary instance, model, and assigned-node identifiers remain unchanged; an
identifier exceeding 256 UTF-8 bytes is represented as a stable
`sha256:<digest>` reference. The assigned-node list retains at most 64 entries
and rejects non-string values during strict replay, so hostile explicit
placement input cannot inflate replicated state and corrupted snapshots cannot
invent authoritative identities.
Skulk records the entry before removing the failed instance, so operators can
distinguish a runner crash, unresponsive or wedged runner, trust rejection,
unrecoverable placement, or lost node from a clean operator stop. Prompt and
generated-response content never enters this history. Replacing the same model
creates a new instance identity and does not erase the earlier failure.

The response also carries a derived `nodeHealth` map (keyed by node id) so a
problem on a node is visible rather than silent. Each entry is a `level`
(`ok`, `warn`, or `error`) plus a list of `reasons`, where each reason has a
`code`, a `message` describing what is wrong, and a `remediation` describing how
to fix it. It is computed read-only from state already in the response (terminal
download failures, low or full models-volume disk, and late liveness signals),
so it adds no new polling. Liveness uses the freshest of the dedicated
telemetry heartbeat, ordinary telemetry fallback, and `lastSeen`. The
`lastSeen` response field is only the last indexed control-plane event and may
be stale for a healthy node; it must not be interpreted as a heartbeat. A node
with no problems reports `level: "ok"` with an empty `reasons` list.

When known live node identities report different Skulk package versions or
source commits, every topology entry receives the warning-level
`version_mismatch` reason. This marks a staggered deployment as degraded until
all nodes converge. Operational visibility remains available, but events,
commands, state, and inference are not cross-version-compatible; finish the
deployment before starting new inference work.

The response carries a live `nodeResources` map as well. Each node entry includes
its placement `backends`, declared `participation`, `apiAvailable` (whether the
node process exposes the HTTP API), resolved `dataTransport` (`gossipsub` or
`zenoh`), `zenohConnectedPeers` (the node's live Zenoh
peer-transport count, sampled at each advertisement; `null` when the node runs
gossipsub or while the count is not yet trustworthy after startup), and
`capabilityConflicts`: loud
observation-vs-declaration disagreements from backend derivation, each with a
`code`, `message`, and `remediation`. Conflicts also surface as `nodeHealth`
reasons on the same response: `gpu_serving_disabled` (error level: a visible
GPU that no engine would use, so serving would run far below hardware speed),
and the warning-level `gpu_detection_degraded` (an NVIDIA device present but
not fully detectable), `invalid_engine_binary` (an engine binary override
pointing at an unusable path), and `backend_override_conflict` (a declared
backend the observed hardware cannot support; the declaration is still
honored). A live fleet that advertises both transports receives
the error-level `data_transport_mismatch` reason in every `nodeHealth` entry.
Mixed DATA transports are unsupported: the signal is diagnostic and does not
bridge traffic. Configure and restart every node uniformly before serving
inference. A node advertising Zenoh with a trustworthy peer-transport count of
exactly 0 while at least one other live node also advertises Zenoh receives
the error-level `zenoh_isolated` reason: its control plane looks healthy but
every remote model or provider stream through it will fail. The typical cause
is a node that cannot reach peers via local multicast (for example one joined
over a routed or overlay network); the remediation is an explicit
`SKULK_ZENOH_CONNECT` peer endpoint plus a dialable `SKULK_ZENOH_LISTEN`
address. The API includes fresh telemetry-only management nodes, local or
remote, even when replicated worker membership does not carry their entries.
For mixed-version state that predates this field, a missing `apiAvailable`
decodes conservatively as `true`; current `--no-api` workers advertise `false`
explicitly.

The `topology` map lists each node's connections. A socket edge carries the
peer's `sinkMultiaddr` plus a boolean `session` annotation distinguishing its
two sources: `session: false` (the default) marks an HTTP-probe-verified
advertised address, which is dialable and eligible as a placement host, while
`session: true` marks a live, authenticated fabric connection recorded as a
path in its own right. Session edges are what keep a NAT'd or proxied remote
member visible and placeable when none of its advertised addresses are
reachable; their recorded address is the connection's observed remote
endpoint, so placement host selection never uses them as a dial target.
Consumers rendering or analyzing the graph should treat `session` edges as
proof of connectivity, not as routable addresses.

Operational note:

- a follower may briefly report a local view that is behind the elected master
  while it is catching up
- on newer builds, catch-up can start from a snapshot plus retained replay tail
  instead of always rebuilding from event `0`
- if your cluster is mixed-version during rollout, upgrade all nodes before you
  rely on bounded replay retention on the master; an older restarted node may
  not be able to fully resync after old history has been compacted away

### Event log

**GET** `/events`

Returns stored events from the API-side event log.

### Diagnostics

- `GET /v1/diagnostics/node`
- `GET /v1/diagnostics/telemetry`
- `GET /v1/diagnostics/performance-envelopes`
- `GET /v1/diagnostics/performance-envelopes/cluster`
- `POST /v1/diagnostics/node/capture`
- `POST /v1/diagnostics/node/runners/{runner_id}/cancel`
- `GET /v1/diagnostics/cluster`
- `GET /v1/diagnostics/cluster/timeline`
- `GET /v1/diagnostics/cluster/{node_id}`
- `POST /v1/diagnostics/cluster/{node_id}/capture`
- `POST /v1/diagnostics/cluster/{node_id}/runners/{runner_id}/cancel`

Use these endpoints when a node appears stuck loading, warming up, decoding, or
shutting down and you need a read-only snapshot without SSHing into every node.

Behavior notes:

- `GET /v1/diagnostics/telemetry` takes no parameters and returns aggregate
  metrics for the API node's isolated telemetry transport: fixed admission and
  network-queue capacities, current and maximum depth, offered/coalesced/dropped
  readings, successful publishes, publish failures and bytes, no-peer publish
  count (`noPeerPublishes`: publishes that found no peers subscribed on the
  telemetry protocol; sustained growth on a connected node means its
  heartbeats reach nobody and it will not appear in membership, typically a
  build/wire mismatch), plus oldest pending and last-successful-publish age. It never returns telemetry payloads, node/model maps,
  or completed attempt identifiers. Query each node directly for its local
  counters; this endpoint is deliberately separate from the node diagnostics
  bundle so additive telemetry instrumentation does not change that bundle's
  rolling-window schema.
- `GET /v1/diagnostics/performance-envelopes` returns this node's observe-only
  performance envelopes: for each `(hardware class, model, engine+backend,
  quantization)` it has served, a throughput-and-latency-versus-concurrency
  curve. Each envelope lists per-concurrency buckets (request count, mean/p50
  decode tokens/second, aggregate decode tokens/second, p50/p90
  time-to-first-token) and a simple `kneeConcurrency` estimate: the concurrency past which
  aggregate throughput stops rising. It is data only (no serving behavior is
  driven from it), kept in bounded memory, and never touches State, the event
  log, or the telemetry gossip plane. Concurrency is the serving instance's own
  in-flight load when a generation began: the served engines (llama.cpp server,
  vLLM) report their true in-flight count, so the curve is accurate across
  replicas and when several API nodes drive one instance; the single-stream
  engines report none and fall back to this API node's outstanding-request count.
  `GET /v1/diagnostics/performance-envelopes/cluster` fans out to every
  reachable member and returns each one's report, with unreachable members
  listed as explicit failures. The dashboard's Performance tab renders these.
- `GET /v1/diagnostics/node` returns the local node's runtime/config facts,
  resources, process tree, live runner-supervisor state, flight-recorder phase
  state, placement analysis, a bounded `doctor` array, and `dataPlane` plus
  `provider` blocks. Each doctor entry contains `checkId`, `title`, `verdict`,
  `detail`, `consequence`, `remediation`, and `fixAvailable` from the node-local
  doctor registry. The array is capped at 64 entries. The proxied
  `GET /v1/diagnostics/cluster/{node_id}` response carries the same complete
  bundle, including doctor results for the selected node. DATA diagnostics include
  transport/reorder mode; active and terminal lifecycle counts; first-byte and
  stream-span timing; duplicate, reordered, skipped, late, idle-timeout,
  transport-failure, and missing-lifecycle counters; plus router egress queue
  depth, independent command-queue count, per-owner pressure, drops, publish
  failures, idle stream reclamations, byte volume, and enqueue/publish latency.
  `dataPlane.egress.idleStreamReclaims` and each owner's matching counter
  increase when a remote command queue emits no frame for its 30-minute resource
  lease and is forcibly terminated and released. The dashboard Node tab renders
  the operational subset and highlights non-zero failure counters.
  Provider diagnostics report active unary calls and streams, concurrency
  limits and high-water marks, admissions and overload rejections, caller input
  queue depth, input/output frame and inline-media byte volume, first-output and
  total stream latency, terminal outcomes, cancellation requests, and
  missing-terminal streams. The same counters are grouped by qualified
  capability ID without retaining call IDs, audio, transcripts, or payloads.
- `POST /v1/diagnostics/node/capture` collects an on-demand local diagnostic
  bundle. Body fields are `runnerId`, `taskId`, `includeProcessSamples`, and
  `sampleDurationSeconds`; all are optional. When a runner/task is provided,
  the response includes that runner's bounded flight recorder, latest MLX
  memory snapshot, and best-effort macOS `sample`, `vmmap -summary`, and
  `footprint -p` output. Sampling failures are returned as structured partial
  failures instead of failing the bundle.
- `POST /v1/diagnostics/node/runners/{runner_id}/cancel` requests cooperative
  cancellation for one task that the local runner supervisor still knows about.
- `GET /v1/diagnostics/cluster` fans out to reachable peer APIs and returns
  partial results when some peers are unavailable. The sweep uses a fail-fast
  probe budget (single attempt, short timeout per advertised address) so one
  unroutable address cannot stall the response. Every topology member appears
  in `nodes`: peers with no reachable API route are explicit `ok: false`
  entries with a `no reachable API route` error rather than being omitted, so
  an overlay-joined node always has an observability presence. Peer diagnostic
  reads ignore unknown additive fields recursively and use compatibility
  defaults for additive counters. The response returns aggregate
  `versionStatus` (`consistent`, `mixed`, or `unknown`) and per-node
  `versionStatus` (`current`, `version_mismatch`, or `unknown`). This tolerance
  applies only to operational diagnostics, not correctness-bearing wire types.
- `GET /v1/diagnostics/cluster/timeline` stitches every reachable node's
  runner-supervisor diagnostics into one cross-rank chronological view. The
  response carries a per-runner synopsis sorted by `(modelId, deviceRank)`
  and every flight-recorder entry across all ranks merged and sorted by `at`.
  Use this when debugging a distributed deadlock: the rank-disagreement
  signature ("rank 0 entered `pipeline_last_eval_output` at T while rank 1
  is still in `pipeline_first_recv_like`") is invisible from any single
  node's local diagnostics but obvious top-to-bottom in the merged timeline.
  Unreachable peers are returned in `unreachableNodes` instead of failing
  the request.
- `GET /v1/diagnostics/cluster/{node_id}` proxies one reachable peer bundle or
  returns the local bundle if `node_id` is the current API node.
- `POST /v1/diagnostics/cluster/{node_id}/capture` proxies the same on-demand
  capture request to a reachable peer node.
- `POST /v1/diagnostics/cluster/{node_id}/runners/{runner_id}/cancel` proxies
  the same cooperative live-runner cancellation request to a reachable peer.
- Placement diagnostics explicitly include whether the current master is part of
  each model placement, which helps investigate hangs where the master is not
  one of the inference ranks.
- The dashboard node inspect icon uses these endpoints to open live diagnostics
  for any reachable node. DATA pressure appears in the `DATA Plane` section.
- The diagnostics drawer prefers `Capture bundle` before cancellation so
  operators can collect phase, MLX memory, and process samples before changing
  the runner state.
- Runner cancellation is best-effort only. A wedged native/MLX runner may
  ignore the request and still require stronger intervention.
- Diagnostics endpoints do not currently kill or restart runners. Capture is
  read-only; the only mutating diagnostics action is the cooperative task-cancel
  request above.

Example:

```bash
curl http://localhost:52415/v1/diagnostics/node
curl http://localhost:52415/v1/diagnostics/telemetry
curl http://localhost:52415/v1/diagnostics/cluster
curl http://localhost:52415/v1/diagnostics/cluster/timeline
curl http://localhost:52415/v1/diagnostics/cluster/<node_id>
curl -X POST http://localhost:52415/v1/diagnostics/node/capture \
  -H 'content-type: application/json' \
  -d '{"runnerId":"<runner_id>","taskId":"<task_id>"}'
curl -X POST http://localhost:52415/v1/diagnostics/cluster/<node_id>/capture \
  -H 'content-type: application/json' \
  -d '{"runnerId":"<runner_id>","includeProcessSamples":true}'
curl -X POST http://localhost:52415/v1/diagnostics/node/runners/<runner_id>/cancel \
  -H 'content-type: application/json' \
  -d '{"taskId":"<task_id>"}'
curl -X POST http://localhost:52415/v1/diagnostics/cluster/<node_id>/runners/<runner_id>/cancel \
  -H 'content-type: application/json' \
  -d '{"taskId":"<task_id>"}'
```

### Field telemetry

- `GET /v1/telemetry/preview`

**GET** `/v1/telemetry/preview`

Returns the field-telemetry consent state and the exact pending sample batch
that would next be sent to the ingest service, so operators can inspect
precisely what leaves the cluster before or after opting in. Collection is
opt-in (dashboard consent flow; `telemetry:` in `skulk.yaml`) and
content-free: samples carry model ids, canonical hardware classes, timing,
token counts, and failure-class enums only. No parameters.

```json
{
  "enabled": false,
  "consent": "unasked",
  "pending": [],
  "dropped_since_start": 0,
  "install_id": "",
  "ingest_url": "https://..."
}
```

### Traces

- `GET /v1/tracing`
- `PUT /v1/tracing`
- `GET /v1/traces`
- `GET /v1/traces/cluster`
- `POST /v1/traces/delete`
- `GET /v1/traces/{task_id}`
- `GET /v1/traces/{task_id}/stats`
- `GET /v1/traces/{task_id}/raw`
- `GET /v1/traces/cluster/{task_id}`
- `GET /v1/traces/cluster/{task_id}/stats`
- `GET /v1/traces/cluster/{task_id}/raw`

Use these endpoints when you are debugging generation behavior, cluster execution, or performance.

Behavior notes:

- `GET /v1/tracing` returns whether runtime tracing is currently enabled for new
  requests across the live cluster session.
- `PUT /v1/tracing` toggles tracing cluster-wide for new requests only. It does
  not retroactively trace in-flight work.
- `GET /v1/traces*` reads local trace artifacts stored on the current node.
- `GET /v1/traces/cluster*` fans out to reachable peer APIs, deduplicates by
  `task_id`, and proxies read-only trace access from any reachable node.
- `POST /v1/traces/delete` remains local-only in v1 even when cluster browsing
  is enabled.

### Runtime tracing control

**GET** `/v1/tracing`

Returns the current cluster tracing state:

```json
{"enabled": false}
```

**PUT** `/v1/tracing`

Enable or disable tracing for new requests across the current cluster session.

Request body:

```json
{"enabled": true}
```

Response body:

```json
{"enabled": true}
```

Operational notes:

- this is a runtime toggle, not a restart-required config edit
- it applies to new requests only
- it does not retroactively trace work already in flight
- the dashboard traces page uses this same API

### Local trace endpoints

These endpoints operate on trace artifacts stored on the current node:

- `GET /v1/traces` lists local trace artifacts with metadata such as task kind,
  model, source nodes, and tool-activity tags
- `GET /v1/traces/{task_id}` returns structured trace events for one task
- `GET /v1/traces/{task_id}/stats` returns aggregated timing summaries
- `GET /v1/traces/{task_id}/raw` downloads Chrome-trace-compatible JSON
- `POST /v1/traces/delete` deletes one or more local trace artifacts

Example:

```bash
curl http://localhost:52415/v1/traces
curl http://localhost:52415/v1/traces/<task_id>/stats
curl -OJ http://localhost:52415/v1/traces/<task_id>/raw
```

### Cluster trace endpoints

These endpoints let a dashboard or script on any reachable node browse traces
across the cluster:

- `GET /v1/traces/cluster`
- `GET /v1/traces/cluster/{task_id}`
- `GET /v1/traces/cluster/{task_id}/stats`
- `GET /v1/traces/cluster/{task_id}/raw`

Operational notes:

- cluster browsing is read-only in v1
- the API fans out to reachable peer APIs and deduplicates traces by `task_id`
- if some peers are unreachable, cluster results may be partial
- source node metadata in responses tells you which nodes contributed trace content

Example:

```bash
curl http://localhost:52415/v1/traces/cluster
curl http://localhost:52415/v1/traces/cluster/<task_id>/stats
curl -OJ http://localhost:52415/v1/traces/cluster/<task_id>/raw
```

## Extension Capabilities

Providers with a dynamic readiness facet appear in capability discovery only
while ready. A cached descriptor does not grant continued admission: new unary
and streaming calls to an unavailable capability return typed `not_found`.
Readiness is rechecked after asynchronous stream admission. Already admitted
calls retain their normal deadline and cancellation behavior.


### List a node's served capabilities

```
GET /v1/capabilities
GET /v1/capabilities?node_id=<id>
```

Returns the self-describing capability descriptors served by a node's
provider extensions (see [Extensions](extensions)). Without `node_id` it
describes the node serving the request; with a peer's `node_id` it proxies
that peer's describe surface (empty when the peer is unreachable). Each
descriptor carries the capability `id`, semantic `version`, a human/LLM-readable
description, JSON Schemas for input and output, the call's I/O mode, and the
response maps each `id@version` to a content revision digest so callers can pin
the exact shape they discovered. Production nodes also include first-party
provider descriptors, including the mounted-model speech providers and stable
`vad@1.0.0`, so descriptor presence alone is not always a liveness claim.
Extensions consume this through
`describe_node`; the light discovery layer (which nodes offer which capability
tag) rides the telemetry plane and appears as `nodeCapabilities` in
`GET /state`.

### Invoke a capability on this node

```
POST /v1/capabilities/call
```

Dispatch one unary capability call to this node's provider extensions. The
body is the typed call envelope: `call_id`, `capability_id`, exact `version`,
the `descriptor_revision` pinned at discovery, `caller_node`, `target_node`,
`timeout_seconds`, and the opaque `payload`. The payload is validated against
the descriptor's input schema before the provider runs, the result against
its output schema after, and payloads are capped at 1 MiB in each direction.
A syntactically valid envelope always gets HTTP 200 with a typed result
(`call_id`, `ok`, `result`, `error`); failures arrive as machine-readable
codes (`not_found`, `version_mismatch`, `revision_mismatch`,
`invalid_payload`, `invalid_result`, `payload_too_large`, `overloaded`,
`timeout`, `provider_error`), so callers switch on `error.code` rather than
transport status. A body that does not parse as the envelope at all
(malformed JSON, missing fields, out-of-range values) gets the standard 422,
since there is no call id to correlate a typed result to. Extensions
normally use this through their context's `call_capability` rather than
calling the endpoint directly.

### Open a streaming capability on this node

```
POST /v1/capabilities/stream
```

This is the control-sized node-to-node opening verb for a provider descriptor
whose `io_mode` is `server_streaming`, `client_streaming`, or `bidirectional`.
The request body is the same pinned
`CapabilityCall` envelope used by unary calls. Skulk checks target identity,
handler/version/revision, the 1 MiB request limit, the descriptor's input
schema, the per-node stream concurrency bound, and the single deadline budget.
Providers may then perform dynamic admission, such as checking that a requested
model is mounted and healthy, inside those same bounds and before lifecycle
creation.
It then returns a typed `CapabilityResult`: `ok: true` with
`{"admitted": true, "io_mode": "..."}` means the stream was admitted; a
pre-admission rejection uses the same typed call errors as the unary endpoint
and creates no stream.

Output is **not** an HTTP response stream. After admission, the provider emits
`started`, ordered `chunk` frames, and exactly one `completed`, `failed`, or
`cancelled` terminal on the provider DATA topic. The handler must return after
yielding that terminal; Skulk withholds it until iterator exhaustion so handler
cleanup finishes before dependent calls can observe completion. Malformed or
trailing handler output closes a closable iterator before Skulk publishes its
synthetic failure terminal. Structured frame metadata is JSON-schema validated
against `output_chunk_schema`; realtime media is an optional raw binary
attachment capped at 1 MiB per frame, while large immutable results use staged
blob references. The topic is node-addressed to
`caller_node`, short-circuits same-node calls, and uses the DATA plane's bounded
per-owner/call/direction Zenoh queues for remote calls. Extensions consume the
flow through `ExtensionContext.stream_capability(...)`, which returns a
`CapabilityStreamSession` containing the typed opening result, one output
iterator, and an `input` sink only for client-streaming/bidirectional calls.
`input.send_chunk()` moves structured metadata plus optional raw media to the
provider; `input.complete()` half-closes caller input without closing provider
output. Input cancellation, invalid schema, unresolved sequence gaps, and queue
pressure produce a typed terminal for only that call.

### Cancel an admitted capability stream

```
POST /v1/capabilities/stream/cancel
```

Accepts `call_id`, `caller_node`, `target_node`, and an optional cancellation
message. Only the caller identity that opened the active stream can cancel it.
Cancellation is idempotent: an active handler is cancelled and emits one typed
`cancelled` terminal; an already-terminal or unknown call returns
`{"cancelled": false}`. `stream_capability` sends this request automatically
when its iterator closes before a terminal frame.

### Stream speech through the built-in TTS provider

Production nodes describe a first-party `tts@1.0.0` server-streaming
capability. It is a facade over core `mlx_audio` serving, not a second model
runtime: the provider translates the generic call into the existing mounted
model `SpeechSynthesis` command and translates `AudioChunk` output into raw
binary provider media frames.

Its payload accepts:

| Field | Type | Meaning |
|-------|------|---------|
| `model` | string | Required mounted TTS model id |
| `text` | string | Required non-empty text to synthesize |
| `response_format` | string | Optional; only `mp3` is accepted in version 1 |
| `voice` | string | Optional model-specific voice |
| `streaming_interval` | number | Optional positive generation cadence hint |
| `speed`, `instruct`, `lang_code` | model-specific | Optional speech controls |
| `temperature`, `top_p`, `repetition_penalty` | number | Optional model-specific sampling controls |
| `top_k`, `max_tokens`, `seed` | integer | Optional model-specific sampling controls; `seed` must be unsigned 32-bit |

Each `chunk` payload reports `model`, `format: "mp3"`, `chunk_index`,
`is_partial`, and optional `sample_rate`; the MP3 bytes are carried beside it
as an `InlineMediaAttachment` with `media_type: "audio/mpeg"`.

The descriptor is always available for contract discovery, while the `tts`
telemetry tag is advertised when at least one eligible model is mounted and
every routable instance of an eligible model has a ready runner.
Dynamic admission rechecks the requested model before `started`. A
caller cancellation propagates to the underlying synthesis command.

### Transcribe a bounded clip through the built-in STT provider

Production nodes describe a first-party `stt@1.0.0` client-streaming
capability and advertise its `stt` telemetry tag while a ready, single-host STT
runner is mounted. The operation is batch inference: client streaming is used
only so encoded audio remains binary provider media instead of base64 in the
unary JSON envelope.

The opening payload requires `model` and optionally accepts `filename`,
`content_type`, `language`, `prompt`, `temperature`, `max_tokens`,
`chunk_duration`, `frame_threshold`, `context`, `prefill_step_size`, `text`,
`word_timestamps`, and `timestamp_granularities`. Send the complete clip as one
or more ordered `InlineMediaAttachment` values, each at most 1 MiB and at most
25 MiB in aggregate, then call the input sink's `complete()` method. Empty,
oversized, cancelled, or non-inline input fails only that provider call.

After input half-close, Skulk runs the existing mounted-model
`AudioTranscription` path. The provider emits no partial transcript chunks; its
single `completed` payload contains `model`, `text`, and optional `language`
and `segments`. Managed blob references are not accepted until Skulk has a
general immutable blob service.

### Transcribe realtime PCM through the built-in STT provider

Production nodes also describe a first-party `stt.realtime@1.0.0`
bidirectional capability. It is advertised only when eligible mounted capacity
is ready and reachable. The owning API may differ from the speech runner node:
same-node input short-circuits locally, while remote input requires the
node-addressed Zenoh data plane. Remote capacity is not advertised when Zenoh
is unavailable.

The opening payload accepts:

| Field | Type | Meaning |
|-------|------|---------|
| `model` | string | Required mounted realtime STT model id |
| `sample_rate` | integer | Input PCM sample rate from 8000 through 96000 Hz |
| `temperature` | number | Optional decode temperature; defaults to `0` |
| `transcription_delay_ms` | integer | Optional upstream cadence from 80 through 2400 ms, in 80 ms steps; defaults to `480` |

Each caller `chunk` must carry an `InlineMediaAttachment` containing mono,
signed little-endian 16-bit PCM. Its payload and attachment metadata must agree
on `format: "pcm_s16le"`, `sample_rate`, and `channels: 1`. The input sink's
`complete()` method half-closes audio input and lets final decoding finish.
Provider output `chunk` frames contain `model`, transcript `text`, and
`is_partial: true`; the `completed` payload contains the accumulated final text
with `is_partial: false`. Skulk withholds that terminal until core cleanup has
sent `TaskFinished` and replicated state reports the task terminal or deleted,
so a following turn cannot be rejected against stale busy state.

Admission pins a `RealtimeAudioTranscription` task to one selected single-host
model instance, and the master reserves that instance against concurrent
admission. Audio bypasses event-sourced State and travels through bounded raw
PCM packets to the serving worker plus a bounded worker-to-runner channel; only
transcript output uses the existing core DATA lifecycle. The mounted upstream
model must expose a true `create_streaming_session` interface. Batch STT cards
are never promoted to realtime by buffering a complete recording.

### Detect speech turns through the built-in VAD provider

Every production API advertises `vad@1.0.0`. Open a bidirectional capability
stream with `sample_rate` set to 8000, 16000, 32000, or 48000 and send ordered
mono `pcm_s16le` inline media. Optional settings are `aggressiveness` (0-3),
`frame_ms` (10, 20, or 30), `minimum_speech_ms`, `silence_hangover_ms`,
`preroll_ms`, and `maximum_utterance_ms`; the descriptor publishes their exact
bounds. Output chunks contain `event` (`speech_started` or `speech_stopped`),
`timestamp_ms`, `reason`, and `preroll_ms`. The completed payload reports the
turn count. Input must end on an exact classifier-frame boundary. Media is
processed within the call and is not retained.

### Realtime transcription WebSocket compatibility edge

```
WS /v1/realtime?model=<mounted-realtime-stt-model>
```

This transcription-only WebSocket is an API-edge adapter over the same
`stt.realtime@1.0.0` provider described above. It does not own model placement,
runner sessions, or a second speech implementation. The API node accepting the
socket owns the provider call, which may select a speech runner on another node.
The same truthful-card, runner-readiness, and Zenoh remote-capacity gates apply.
OpenAPI does not model WebSocket operations, so this manual section is the
normative edge contract;
the underlying provider opening remains represented by the documented HTTP
capability endpoints.

The wire contract implements a bounded subset of OpenAI Realtime transcription:

| Direction | Event | Behavior |
|---|---|---|
| server to client | `session.created` | Reports a `type: transcription` session with the selected model and fixed PCM input configuration. |
| client to server | `session.update` | Confirms the current nested `audio.input` configuration. `turn_detection` may be null or a bounded `server_vad` configuration. Optional `response` selects a mounted chat `model`, optional mounted `tts_model`, optional `voice`, `max_output_tokens` from 1 through 4096 (default 256), and `enable_thinking` (default false); attempts to change the input model/codec, enable noise reduction/language hints, or add unsupported fields are rejected. |
| server to client | `session.updated` | Confirms an accepted current session update. |
| client to server | `input_audio_buffer.append` | Appends one base64 PCM16 frame and immediately forwards its decoded bytes as binary Fabric media. |
| client to server | `input_audio_buffer.commit` | Half-closes the current utterance and triggers final provider drain. Empty commits and duplicate manual commits are rejected. A manual commit racing after server VAD has already auto-committed the same utterance is an idempotent no-op. The next turn may begin after its completed event. |
| server to client | `input_audio_buffer.speech_started` | Reports the detected start timestamp and current item when server VAD is enabled. |
| server to client | `input_audio_buffer.speech_stopped` | Reports the detected end timestamp immediately before server VAD commits the utterance. |
| server to client | `input_audio_buffer.committed` | Confirms the input half-close. |
| server to client | `conversation.item.input_audio_transcription.delta` | Carries one provider transcript delta after commit. |
| server to client | `conversation.item.input_audio_transcription.completed` | Carries the accumulated final transcript, completes the current item, and leaves the socket ready for another turn. |
| server to client | `conversation.item.input_audio_transcription.failed` | Carries a provider/transport/cancellation terminal failure. |
| server to client | `response.created` | Announces automatic assistant work after a final transcript when `session.response` is configured. |
| server to client | `response.output_text.delta` / `response.output_text.done` | Streams visible assistant text and its bounded final value. Reasoning tokens and tool calls are not exposed or synthesized. |
| server to client | `response.audio.delta` / `response.audio.done` | Streams base64 MP3 chunks from the selected mounted `tts_model`. |
| client to server | `response.cancel` | Cancels active model generation or TTS. New speech detected by server VAD performs the same cancellation before starting the next turn. |
| server to client | `response.done` | Terminates one assistant response with `completed`, `cancelled`, or `failed` status. |
| server to client | `error` | Reports invalid client events, unsupported configuration, or response failures. Policy and transport errors may close the socket; response failures are non-terminal to the socket and are followed by `response.done`. |

Version 1 accepts JSON text WebSocket messages and base64-encoded mono,
signed little-endian PCM16 at 24 kHz. A decoded audio frame is capped at 1 MiB,
the encoded WebSocket event at 2 MiB, and one session at 64 MiB of decoded
audio. Provider transcript text is capped at 1 MiB per event and in the
pre-commit buffer; overflow emits a typed transcription failure and closes the
socket with `1011`. `input_audio_buffer.clear` is deliberately unsupported because the API
forwards audio incrementally and retains no replay buffer that could safely
retract already-delivered media. Browser connections must be same-origin; SDK
clients without an `Origin` header remain supported.

`turn_detection: {"type":"server_vad"}` enables server-owned WebRTC VAD.
Optional settings are `aggressiveness` (0-3), `prefix_padding_ms` (0-2000),
`silence_duration_ms` (20-5000), `minimum_speech_ms` (20-5000), and
`maximum_utterance_ms` (100-120000). The edge incrementally resamples the
24 kHz input to the classifier's 16 kHz frame contract, emits typed speech
boundaries, and commits on silence or the maximum utterance duration. The edge
forwards VAD-enabled input in 20 ms source-rate slices and stops at the
detected boundary, so the unprocessed remainder of a large append cannot leak
into the committed utterance. The socket serializes turns: each utterance opens
one bounded Fabric provider call,
and audio appended while a committed turn is still draining receives a
non-terminal `turn_in_progress` error. Completed turns rotate `item_id`, link
the next commit through `previous_item_id`, reset VAD state, and release their
provider capacity. The 64 MiB decoded-audio bound applies across the complete
WebSocket session.

The dashboard chat microphone uses this edge only when both the selected model
declares streaming/realtime audio and the API node currently advertises the
stable `stt.realtime` provider. An `AudioWorklet` captures mono browser
samples, the dashboard continuously resamples them to 24 kHz PCM16, and the
client aggregates worklet callbacks into 100 ms transport frames before the
mic control commits the socket when recording stops. Realtime mode can retain
the socket across server-VAD turns, show partial transcripts in the editable
draft, and optionally auto-send final transcripts through the same dashboard
`POST /v1/chat/completions` flow used by typed prompts. This preserves the full
dashboard conversation and its ordinary generation, cancellation, and TTS
semantics; the WebSocket's optional `response` participant remains available to
API clients but is not a second dashboard conversation. If either capability
truth is absent, chat retains the batch `MediaRecorder` plus
`POST /v1/audio/transcriptions` path.

When `response` is configured, the API node that owns the WebSocket retains the
bounded text-only conversation history for that socket, routes each final
transcript through the selected mounted chat model with the configured
`max_output_tokens` ceiling. Hidden reasoning is disabled by default so the
bounded budget produces speech-ready visible text; clients may opt in with
`enable_thinking`. The edge then optionally opens a normal `tts@1.0.0` Fabric
provider stream for the visible final answer. Explicit
`response.cancel`, a new non-VAD audio turn, or VAD speech detection cancels the
active model/TTS command before the replacement turn proceeds. Media bytes are
not added to conversation history or State.

The edge does not implement noise reduction, G.711, ephemeral session-token
creation, client-created conversation items, or tool execution.
Provider capacity failures close with retryable WebSocket code `1013`; client
protocol/policy violations use `1003`, `1008`, or `1009`; internal provider
failures use `1011`. Disconnecting before a terminal event cancels the provider
input and output directions.

For compatibility with clients written against the earlier transcription beta,
the edge also accepts `transcription_session.update` with
`input_audio_format`, `input_audio_transcription`, `turn_detection`, and
`input_audio_noise_reduction`, replying with `transcription_session.updated`.

### Compose a typed Fabric speech chain

**WS** `/v1/fabric/chains/speech?stt_model=<mounted-realtime-stt-model>`

This first-class composition surface uses the same hardened event contract as
`/v1/realtime`, but names the endpoint by its Fabric role. After
`session.created`, send `session.update` to select server VAD and an optional
`response` containing mounted `model`, `tts_model`, and `voice` participants
plus optional bounded `max_output_tokens` and `enable_thinking` controls.
Input PCM, transcript events, assistant text, TTS audio, cancellation, bounded
history, and terminal status retain the contracts documented above.

The chain resolves every participant through normal mounted capability and
health checks. It does not create a second runtime, persist audio or transcripts
in State, perform graph search, or introduce prompt-level authority. Remote
participants continue to use the normal bounded provider data plane, and socket
disconnect or `response.cancel` reaches the active provider/model commands.

```bash
curl http://localhost:52415/v1/capabilities
```

```json
{
  "node_id": "12D3KooW...",
  "capabilities": [
    {
      "id": "echo",
      "version": "1.0.0",
      "title": "Echo",
      "description": "Returns the input text unchanged.",
      "input_schema": {"type": "object", "properties": {"text": {"type": "string"}}},
      "output_schema": {"type": "object"},
      "io_mode": "unary",
      "input_chunk_schema": null,
      "output_chunk_schema": null,
      "annotations": null
    }
  ],
  "revisions": {"echo@1.0.0": "5a1c9e327b6f4d08"}
}
```

## Connectivity Endpoints

### Tailscale status

```
GET /v1/connectivity/tailscale
GET /v1/connectivity/tailscale?node_id=<id>
```

Returns whether tailscaled is running on a node and, if so, the node's Tailscale IP, hostname, DNS name, and tailnet. All fields except `running` are `null` when tailscaled is not installed or not running.

Pass `node_id` to proxy the request to a specific cluster node. Omit it to query the local node directly. Returns `404` if the target node is not reachable.

**Response fields:**

| Field | Type | Description |
| --- | --- | --- |
| `running` | boolean | `true` when tailscaled reports `BackendState == "Running"` |
| `selfIp` | string \| null | Node's Tailscale IPv4 address (100.x.x.x range) |
| `hostname` | string \| null | Node hostname as registered in the tailnet |
| `dnsName` | string \| null | Fully-qualified Tailscale MagicDNS name, e.g. `my-node.tailnet-abc.ts.net` |
| `tailnet` | string \| null | Tailnet name derived from `dnsName` |
| `version` | string \| null | Tailscale client version string |

```bash
# Local node
curl http://localhost:52415/v1/connectivity/tailscale

# Specific cluster node
curl "http://localhost:52415/v1/connectivity/tailscale?node_id=<node-id>"
```

### Remote access info

```
GET /v1/connectivity/remote-access
```

Returns aggregated remote access information for the local node: LAN address, Tailscale address, and a `preferredUrl` (Tailscale if running, otherwise LAN). When Tailscale is running, `preferredUrl` uses the node's MagicDNS name (`my-node.tailnet-abc.ts.net`) if available, falling back to the raw `100.x.x.x` IP. `operatorUrl` appends `/operator` to `preferredUrl` (suitable for QR code generation so mobile users land directly on the operator panel).

**Response fields:**

| Field | Type | Description |
| --- | --- | --- |
| `local.ip` | string \| null | Preferred LAN IPv4 address |
| `local.port` | integer | API/dashboard port |
| `local.url` | string \| null | `http://{ip}:{port}` |
| `tailscale.running` | boolean | `true` when tailscaled is connected |
| `tailscale.ip` | string \| null | Tailscale IPv4 address (100.x.x.x) |
| `tailscale.dnsName` | string \| null | MagicDNS fully-qualified name, e.g. `my-node.tailnet-abc.ts.net` |
| `tailscale.port` | integer | API/dashboard port |
| `tailscale.url` | string \| null | `http://{dnsName or ip}:{port}` if running |
| `preferredUrl` | string \| null | MagicDNS URL if available, else Tailscale IP URL, else LAN URL |
| `operatorUrl` | string \| null | `preferredUrl + /operator` |

```bash
curl http://localhost:52415/v1/connectivity/remote-access | python3 -m json.tool
```

Example response when Tailscale is running with MagicDNS:

```json
{
  "local": { "ip": "192.168.1.5", "port": 52415, "url": "http://192.168.1.5:52415" },
  "tailscale": {
    "running": true,
    "ip": "100.101.102.103",
    "dnsName": "my-node.tailnet-abc.ts.net",
    "port": 52415,
    "url": "http://my-node.tailnet-abc.ts.net:52415"
  },
  "preferredUrl": "http://my-node.tailnet-abc.ts.net:52415",
  "operatorUrl": "http://my-node.tailnet-abc.ts.net:52415/operator"
}
```

## Operator App Integration

The operator panel at `/operator` is designed for mobile access and can also be driven by a native app. The relevant API endpoints are:

### Node and cluster state

| Endpoint | Description |
| --- | --- |
| `GET /state` | Full cluster state: nodes, instances, recent terminal instance failures, runners, memory, GPU |
| `GET /node_id` | Local node's ID |
| `GET /node/identity` | Node ID, hostname, and preferred LAN IP |

### Remote access and connectivity

| Endpoint | Description |
| --- | --- |
| `GET /v1/connectivity/remote-access` | LAN + Tailscale addresses, preferred URL, operator URL for QR |
| `GET /v1/connectivity/tailscale` | Tailscale status for local node |
| `GET /v1/connectivity/tailscale?node_id=<id>` | Tailscale status for a specific peer node |

### Node management

| Endpoint | Description |
| --- | --- |
| `POST /admin/restart?node_install_id=<id>` | Resolve a stable installation identity and send a restart command to its current live node |

### Typical operator app workflow

1. Call `GET /v1/connectivity/remote-access` on the initially discovered node to get the `preferredUrl`, then use that as the base URL for subsequent calls.
2. Poll `GET /state` every 5 seconds for node health (memory, GPU, temperature)
   and stable `nodeIdentities[*].nodeInstallId` values.
3. Show restart only when the selected live node reports a stable installation
   identity, then call `POST /admin/restart?node_install_id=<id>`.
4. On first launch or settings screen, show the `operatorUrl` as a QR code so users can hand it off to another device.

## Helpful Next Docs

- [README](https://github.com/Foxlight-Foundation/Skulk/blob/main/README.md)
- [Tracing and debugging](tracing)
- [Model store guide](model-store)
- [Architecture overview](architecture)
- [API Reference](/api/skulk-api)
