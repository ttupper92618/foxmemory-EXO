from typing import Self

import anyio
from anyio import (
    CancelScope,
    Event,
    get_cancelled_exc_class,
)
from loguru import logger

from skulk.routing.connection_message import ConnectionMessage
from skulk.shared.types.commands import ForwarderCommand
from skulk.shared.types.common import NodeId, SessionId
from skulk.utils.channels import Receiver, Sender
from skulk.utils.pydantic_ext import CamelCaseModel
from skulk.utils.task_group import TaskGroup

DEFAULT_ELECTION_TIMEOUT = 3.0

# Connection updates arrive far more often than cluster membership actually
# changes: libp2p re-dials multi-homed peers (link-local + LAN + overlay
# addresses) and pings every few seconds, and a ping timeout briefly tears a
# connection down before it re-establishes. Each update used to restart the
# whole election campaign, and because that churn cadence (~2.5s) is shorter
# than DEFAULT_ELECTION_TIMEOUT, a campaign could be cancelled-and-restarted
# forever and never elect a master, so the cluster never forms. We coalesce a
# burst of related updates over this window before deciding whether membership
# truly changed.
CONNECTION_SETTLE_DELAY = 0.2


class ElectionMessage(CamelCaseModel):
    clock: int
    seniority: int
    proposed_session: SessionId
    commands_seen: int

    # Could eventually include a list of neighbour nodes for centrality
    def __lt__(self, other: Self) -> bool:
        if self.clock != other.clock:
            return self.clock < other.clock
        if self.seniority != other.seniority:
            return self.seniority < other.seniority
        elif self.commands_seen != other.commands_seen:
            return self.commands_seen < other.commands_seen
        else:
            return (
                self.proposed_session.master_node_id
                < other.proposed_session.master_node_id
            )


class ElectionResult(CamelCaseModel):
    session_id: SessionId
    won_clock: int
    is_new_master: bool


class Election:
    def __init__(
        self,
        node_id: NodeId,
        *,
        election_message_receiver: Receiver[ElectionMessage],
        election_message_sender: Sender[ElectionMessage],
        election_result_sender: Sender[ElectionResult],
        connection_message_receiver: Receiver[ConnectionMessage],
        command_receiver: Receiver[ForwarderCommand],
        is_candidate: bool = True,
        seniority: int = 0,
    ):
        # If we aren't a candidate, simply don't increment seniority.
        # For reference: This node can be elected master if all nodes are not master candidates
        # Any master candidate will automatically win out over this node.
        self.seniority = seniority if is_candidate else -1
        self.clock = 0
        self.node_id = node_id
        self.commands_seen = 0
        # Every node spawns as master
        self.current_session: SessionId = SessionId(
            master_node_id=node_id, election_clock=0
        )

        # Senders/Receivers
        self._em_sender = election_message_sender
        self._em_receiver = election_message_receiver
        self._er_sender = election_result_sender
        self._cm_receiver = connection_message_receiver
        self._co_receiver = command_receiver

        # Campaign state
        self._candidates: list[ElectionMessage] = []
        self._completed_election: ElectionMessage | None = None
        self._campaign_cancel_scope: CancelScope | None = None
        self._campaign_done: Event | None = None
        # Live connection count per peer. A multi-homed peer holds several
        # connections at once and the transport emits one update per connection,
        # so we ref-count (mirroring `connected_ips` in the Rust discovery layer)
        # and treat a peer as present while it has at least one connection. Only
        # a peer crossing 0<->1 connections changes membership and may trigger a
        # new election; a duplicate connect or one of several connections closing
        # must not, and the genuine last close must still register as a departure.
        self._connected_peers: dict[NodeId, int] = {}
        self._tg = TaskGroup()

    async def run(self):
        logger.info("Starting Election")
        try:
            async with self._tg as tg:
                tg.start_soon(self._election_receiver)
                tg.start_soon(self._connection_receiver)
                tg.start_soon(self._command_counter)

                # And start an election immediately, that instantly resolves
                candidates: list[ElectionMessage] = []
                logger.debug("Starting initial campaign")
                self._candidates = candidates
                await self._campaign(candidates, campaign_timeout=0.0)
                logger.debug("Initial campaign finished")
        finally:
            # Cancel and wait for the last election to end
            if self._campaign_cancel_scope is not None:
                logger.debug("Cancelling campaign")
                self._campaign_cancel_scope.cancel()
            if self._campaign_done is not None:
                logger.debug("Waiting for campaign to finish")
                await self._campaign_done.wait()
            logger.debug("Campaign cancelled and finished")
            logger.info("Election shutdown")

    async def elect(self, em: ElectionMessage) -> None:
        """Publish the selected session and retain its vote for late-round repair."""
        logger.debug(f"Electing: {em}")
        is_new_master = em.proposed_session != self.current_session
        self.current_session = em.proposed_session
        self._completed_election = em
        logger.debug(f"Current session: {self.current_session}")
        await self._er_sender.send(
            ElectionResult(
                won_clock=em.clock,
                session_id=em.proposed_session,
                is_new_master=is_new_master,
            )
        )

    async def shutdown(self) -> None:
        self._tg.cancel_tasks()

    async def _election_receiver(self) -> None:
        with self._em_receiver as election_messages:
            async for message in election_messages:
                logger.debug(f"Election message received: {message}")
                if message.proposed_session.master_node_id == self.node_id:
                    logger.debug("Dropping message from ourselves")
                    # Drop messages from us (See skulk.routing.router)
                    continue
                # If a new round is starting, we participate
                if message.clock > self.clock:
                    self.clock = message.clock
                    logger.debug(f"New clock: {self.clock}")
                    logger.debug("Starting new campaign")
                    candidates: list[ElectionMessage] = [message]
                    logger.debug(f"Candidates: {candidates}")
                    logger.debug(f"Current candidates: {self._candidates}")
                    self._candidates = candidates
                    logger.debug(f"New candidates: {self._candidates}")
                    logger.debug("Starting new campaign")
                    self._tg.start_soon(
                        self._campaign, candidates, DEFAULT_ELECTION_TIMEOUT
                    )
                    logger.debug("Campaign started")
                    continue
                # Dismiss old messages
                if message.clock < self.clock:
                    logger.debug(f"Dropping old message: {message}")
                    continue
                if message in self._candidates:
                    # During the rolling-compatibility window, peers publish
                    # election status on both the legacy and isolated gossipsub
                    # protocols. Treat the second wire copy as the same vote so
                    # seniority and campaign outcomes remain protocol-agnostic.
                    logger.debug(f"Dropping duplicate election candidate: {message}")
                    continue
                logger.debug(f"Election added candidate {message}")
                # Now we are processing this rounds messages - including the message that triggered this round.
                self._candidates.append(message)
                completed = self._completed_election
                if (
                    completed is not None
                    and completed.clock == message.clock
                    and completed < message
                ):
                    # Subscription readiness and network latency can deliver a
                    # same-round vote after our timeout. A healthy connection
                    # then produces no new membership event to repair two
                    # conflicting masters. Compare the completed vote, not our
                    # post-win seniority, and monotonically adopt a better vote.
                    await self.elect(message)

    def _apply_connection_updates(self, updates: list[ConnectionMessage]) -> bool:
        """Fold connection updates into the per-peer live-connection counts.

        Returns ``True`` if cluster membership genuinely changed: a peer's first
        connection opened (it joined) or its last connection closed (it left).
        Returns ``False`` if every update was a no-op: a second connection
        opening or closing for a peer that stays present either way, or a
        disconnect for a peer we were not tracking. Updates about ourselves are
        ignored. Multi-homed peers hold several connections and the transport
        emits one update per connection, so we ref-count rather than treat the
        first close as a departure; otherwise a real master loss (the genuine
        last close) could be dropped as an untracked-peer disconnect and never
        trigger re-election. Only real membership changes should restart the
        election, so steady connection churn cannot livelock formation.
        """
        before = frozenset(self._connected_peers)
        for update in updates:
            if update.node_id == self.node_id:
                continue
            if update.connected:
                self._connected_peers[update.node_id] = (
                    self._connected_peers.get(update.node_id, 0) + 1
                )
            else:
                remaining = self._connected_peers.get(update.node_id, 0) - 1
                if remaining > 0:
                    self._connected_peers[update.node_id] = remaining
                else:
                    # Last connection closed (or an unbalanced/untracked close):
                    # drop the peer rather than letting the count go negative.
                    self._connected_peers.pop(update.node_id, None)
        return frozenset(self._connected_peers) != before

    async def _connection_receiver(self) -> None:
        with self._cm_receiver as connection_messages:
            async for first in connection_messages:
                # Delay after a connection message so a burst of related updates
                # (multi-homed dials, symmetric setup) coalesces into one
                # decision instead of one campaign restart per socket event.
                await anyio.sleep(CONNECTION_SETTLE_DELAY)
                rest = connection_messages.collect()

                logger.debug(
                    f"Connection messages received: {first} followed by {rest}"
                )
                if not self._apply_connection_updates([first, *rest]):
                    # No net change to cluster membership (a flap or a duplicate
                    # update to a peer we already track). Restarting the election
                    # here is exactly what let connection churn livelock cluster
                    # formation, so we ignore it.
                    logger.debug("Connection update with no membership change; ignoring")
                    continue

                # Let any in-flight campaign finish before starting a new one. A
                # genuinely flapping peer then causes serialized campaigns that
                # each complete (the cluster converges between flaps) rather than
                # an endless cancel-and-restart loop that never elects a master.
                done = self._campaign_done
                if done is not None and not done.is_set():
                    logger.debug(
                        "Awaiting in-flight campaign before connection-driven round"
                    )
                    await done.wait()

                logger.debug(f"Membership changed; current clock: {self.clock}")
                # These messages are strictly peer to peer
                self.clock += 1
                logger.debug(f"New clock: {self.clock}")
                candidates: list[ElectionMessage] = []
                self._candidates = candidates
                logger.debug("Starting new campaign")
                self._tg.start_soon(
                    self._campaign, candidates, DEFAULT_ELECTION_TIMEOUT
                )
                logger.debug("Campaign started")
                logger.debug("Connection message added")

    async def _command_counter(self) -> None:
        with self._co_receiver as commands:
            async for _command in commands:
                self.commands_seen += 1

    async def _campaign(
        self, candidates: list[ElectionMessage], campaign_timeout: float
    ) -> None:
        clock = self.clock

        # Kill the old campaign
        if self._campaign_cancel_scope:
            logger.info("Cancelling other campaign")
            self._campaign_cancel_scope.cancel()
        if self._campaign_done:
            logger.info("Waiting for other campaign to finish")
            await self._campaign_done.wait()

        done = Event()
        self._campaign_done = done
        scope = CancelScope()
        self._campaign_cancel_scope = scope

        try:
            with scope:
                logger.debug(f"Election {clock} started")

                status = self._election_status(clock)
                candidates.append(status)
                await self._em_sender.send(status)

                logger.debug(f"Sleeping for {campaign_timeout} seconds")
                await anyio.sleep(campaign_timeout)
                # minor hack - rebroadcast status in case anyone has missed it.
                await self._em_sender.send(status)
                logger.debug("Woke up from sleep")
                # add an anyio checkpoint - anyio.lowlevel.chekpoint() or checkpoint_if_cancelled() is preferred, but wasn't typechecking last I checked
                await anyio.sleep(0)

                # Election finished!
                elected = max(candidates)
                logger.debug(f"Election queue {candidates}")
                logger.debug(f"Elected: {elected}")
                if (
                    self.node_id == elected.proposed_session.master_node_id
                    and self.seniority >= 0
                ):
                    logger.debug(
                        f"Node is a candidate and seniority is {self.seniority}"
                    )
                    self.seniority = max(self.seniority, len(candidates))
                    logger.debug(f"New seniority: {self.seniority}")
                else:
                    logger.debug(
                        f"Node is not a candidate or seniority is not {self.seniority}"
                    )
                logger.debug(
                    f"Election finished, new SessionId({elected.proposed_session}) with queue {candidates}"
                )
                logger.debug("Sending election result")
                await self.elect(elected)
                logger.debug("Election result sent")
        except get_cancelled_exc_class():
            logger.debug(f"Election {clock} cancelled")
        finally:
            logger.debug(f"Election {clock} finally")
            if self._campaign_cancel_scope is scope:
                self._campaign_cancel_scope = None
            logger.debug("Setting done event")
            done.set()
            logger.debug("Done event set")

    def _election_status(self, clock: int | None = None) -> ElectionMessage:
        c = self.clock if clock is None else clock
        return ElectionMessage(
            proposed_session=(
                self.current_session
                if self.current_session.master_node_id == self.node_id
                else SessionId(master_node_id=self.node_id, election_clock=c)
            ),
            clock=c,
            seniority=self.seniority,
            commands_seen=self.commands_seen,
        )
