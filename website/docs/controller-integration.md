---
title: Controller integration
description: Observe node and model readiness and reconcile exact placements through the Skulk API.
---

A controller can use Skulk's existing HTTP APIs to connect an external resource
lifecycle to model placement. Keep separate records for resource ownership,
cluster membership, installed models, ready runners, and successful inference.
Each proves a different part of the workflow.

This guide describes the Skulk integration boundary. Resource provisioning,
provider credentials, billing, approval policy, and resource deletion belong to
the controller or its separately installed plugin. Skulk's instance API manages
model instances; deleting an instance does not terminate a rented machine.

## Establish node identity and membership

Call `GET /node_id` on the management API and on the compute node's API over an
owner-controlled connection. The response is that API node's runtime ID as a
JSON string. Bind the compute ID to the external resource you acquired, and
recheck the management identity before placing work. A reachable port alone
does not establish which resource or cluster is behind it.

Poll `GET /state` on both endpoints with bounded request deadlines. Check that
both identities appear in `topology.nodes`, and that each endpoint reports live
resource observations for the other node in `nodeResources`. A one-sided view
can reflect incomplete join or interrupted communication. Skulk filters these
resource observations by its live telemetry view; they remain observations,
not a guarantee of future reachability. A controller should attach its own
observation time and expire cached readiness when polling fails.

Use the normal [API authentication](api-guide.md) configuration. Keep management
API credentials scoped to that API origin; do not forward them automatically to
a new compute node, proxy destination, or redirect.

## Select and submit an exact placement

1. Request `GET /instance/previews?model_id=MODEL_ID&node_ids=NODE_ID`.
2. Select an admitted preview (`error` is null), and inspect its complete
   `instance` object. `node_ids` uses subset matching: verify the actual
   `shardAssignments.nodeToRunner` keys if the workload must run on exactly one
   acquired node. Also inspect the model identity, backend compatibility and
   reported context limit. A preview is not a reservation.
3. Persist the chosen instance and its `instanceId` before making an effectful
   request. Submit `POST /instance` with `{"instance": INSTANCE_OBJECT}`.
4. Retain the response's `command_id` and `instance_id`; correlate later state
   and terminal failures with the exact instance identity.

The exact-placement API checks model-card identity, code authorization, and
aggregate available memory. It does not rerun every preview check for topology,
per-node capacity or backend compatibility. A stale preview can therefore be
accepted and fail asynchronously. HTTP acceptance means the command was accepted;
continue reconciling the assigned nodes and terminal failure history while
download and runner startup proceed. See the [placement API](api-guide.md#create-an-instance-from-a-fully-specified-placement)
for the response and refusal contract.

If the HTTP response is lost, inspect `GET /state` for the stored instance ID
and `instanceFailures` before deciding what to do. The API does not document a
client idempotency-key guarantee for `POST /instance`. A controller must not
assume that a timeout means no effect occurred. Persist a submission fence
before sending, use a bounded reconciliation deadline, and retain an explicit
uncertain outcome when neither an instance nor a terminal failure is observed.
An unrelated instance serving the same model does not prove that the submitted
placement succeeded.

## Distinguish download, runner, and inference readiness

`GET /state` combines durable control state with the current telemetry overlay.
Its `downloads` entries are tagged records grouped by node. Match the target
node and immutable downloadable artifact: signed cards use `registryCardId`,
while legacy/custom cards require full card equality. Download state is retained
per node/model and may describe an earlier shard layout of the same cached
artifact. A new runner can reuse that installation without emitting a new
completion record, so layer, rank and world-size differences in a download
record must not reject an otherwise matching artifact. Bind current execution
to the instance's assigned runner and shard through
`shardAssignments.nodeToRunner` and `runnerToShard`.

| Evidence | What it establishes |
| --- | --- |
| `DownloadPending` / `DownloadOngoing` | A download is queued or progressing; bytes are not yet installation readiness. |
| `DownloadCompleted` | The download has completed installation finalization and the selected artifact's installed identity checks. |
| `DownloadFailed` | That attempt failed; correlate its `attemptId` before attributing it to current work. |
| `RunnerReady` / `RunnerRunning` for the assigned runner | The model runner is ready or executing work. |
| Successful ordinary inference through the management API | The tested request completed across the actual serving path. |

Current download attempts include `attemptId`; older retained records may have
null identities. A controller that needs exact attempt attribution should keep
those legacy records as incomplete evidence. Never let a late transient sample
overwrite a terminal outcome from the same attempt. Treat contradictory
observations as uncertainty and refresh them.

For distributed observations, require agreement on the same instance and
download attempt before reporting ready. Download completion alone does not
establish that model loading succeeded. A ready runner alone does not prove
that an ordinary inference request has completed.

Give initial readiness a finite deadline. If previously ready capacity loses
membership or runner readiness, report the loss immediately and allow only a
bounded recovery interval. Keep the original external resource expiry as an
upper bound across process restarts and transport retries.

## Cleanup and troubleshooting

Persist ownership and cleanup intent before starting or cancelling work. Keep
the provider's cleanup obligation independent of the connection that runs
Skulk. Reaping SSH, stopping Skulk, and `DELETE /instance/{instance_id}` each
address local execution or model state; none confirms external resource
deletion or billing cutoff. Reconcile those through the resource provider and
account separately for retained storage.

| Observed state | Next check |
| --- | --- |
| API responds but peer is missing on one side | Compare node IDs, both topology views and live resource observations. |
| Preview refused | Read `error_code`, compatibility details and current node memory; request a new preview after conditions change. |
| Download progresses but never completes | Inspect the exact attempt and node diagnostics; byte progress does not bypass finalization. |
| Download completed but runner never ready | Correlate runner state and `instanceFailures` with the submitted instance ID. |
| Placement response lost | Reconcile the stored instance ID; do not blindly repeat the POST. |
| Controller stops while cleanup is pending | Retain the resource receipt and retry deletion through the independent cleanup owner until absence is confirmed. |

Use [Node Doctor](node-doctor.md) for node-level diagnosis and the
[API guide](api-guide.md#cluster-state) for state inspection.
Store raw provider and process evidence in protected controller storage;
publish sanitized operational outcomes rather than credentials or payloads.


## Plan capacity for an exact model

Before selecting external capacity, call [`GET /models/requirements`](./api-guide.md#read-model-capacity-requirements)
with the requested `model_id` and per-sequence `context_tokens`. It returns the
catalog's effective installed-card binding, core's advisory whole-model memory
estimate, declared disk bytes and backend evidence. Preserve the requested model
and context in approved intent. A null memory estimate means missing sizing
information, not zero memory. Account separately for runtime images, staging and
other disk use, and do not combine multiple GPUs into one capacity figure without
engine support for that layout.

The requirements read reserves nothing. Catalog/store caches can outlive a live
configuration change, and pre-provision estimates do not prove future placement.
Repeat card identity and compatibility checks against the target environment;
then observe exact placement, download, runner and inference evidence as described
above. Provider offers and budget enforcement belong to the external controller.
