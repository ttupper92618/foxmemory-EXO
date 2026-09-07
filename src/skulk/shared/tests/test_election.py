import anyio
import pytest
from anyio import create_task_group, fail_after, move_on_after

from skulk.routing.connection_message import ConnectionMessage
from skulk.shared.election import Election, ElectionMessage, ElectionResult
from skulk.shared.types.commands import ForwarderCommand, TestCommand
from skulk.shared.types.common import NodeId, SessionId, SystemId
from skulk.utils.channels import channel

# ======= #
# Helpers #
# ======= #


def em(
    clock: int,
    seniority: int,
    node_id: str,
    commands_seen: int = 0,
    election_clock: int | None = None,
) -> ElectionMessage:
    """
    Helper to build ElectionMessages for a given proposer node.

    The new API carries a proposed SessionId (master_node_id + election_clock).
    By default we use the same value for election_clock as the 'clock' of the round.
    """
    return ElectionMessage(
        clock=clock,
        seniority=seniority,
        proposed_session=SessionId(
            master_node_id=NodeId(node_id),
            election_clock=clock if election_clock is None else election_clock,
        ),
        commands_seen=commands_seen,
    )


# ======================================= #
#                 TESTS                   #
# ======================================= #


@pytest.fixture(autouse=True)
def fast_election_timeout(monkeypatch: pytest.MonkeyPatch):
    monkeypatch.setattr("skulk.shared.election.DEFAULT_ELECTION_TIMEOUT", 0.1)


@pytest.mark.anyio
async def test_single_round_broadcasts_and_updates_seniority_on_self_win() -> None:
    """
    Start a round by injecting an ElectionMessage with higher clock.
    With only our node effectively 'winning', we should broadcast once and update seniority.
    """
    # Outbound election messages from the Election (we'll observe these)
    em_out_tx, em_out_rx = channel[ElectionMessage]()
    # Inbound election messages to the Election (we'll inject these)
    em_in_tx, em_in_rx = channel[ElectionMessage]()
    # Election results produced by the Election (we'll observe these)
    er_tx, er_rx = channel[ElectionResult]()
    # Connection messages
    cm_tx, cm_rx = channel[ConnectionMessage]()
    # Commands
    co_tx, co_rx = channel[ForwarderCommand]()

    election = Election(
        node_id=NodeId("B"),
        election_message_receiver=em_in_rx,
        election_message_sender=em_out_tx,
        election_result_sender=er_tx,
        connection_message_receiver=cm_rx,
        command_receiver=co_rx,
        is_candidate=True,
    )

    async with create_task_group() as tg:
        with fail_after(2):
            tg.start_soon(election.run)
            # Trigger new round at clock=1 (peer announces it)
            await em_in_tx.send(em(clock=1, seniority=0, node_id="A"))

            # Expect our broadcast back to the peer side for this round only
            while True:
                got = await em_out_rx.receive()
                if got.clock == 1 and got.proposed_session.master_node_id == NodeId(
                    "B"
                ):
                    break

            # Wait for the round to finish and produce an ElectionResult
            result = await er_rx.receive()
            assert result.session_id.master_node_id == NodeId("B")
            # We spawned as master; electing ourselves again is not "new master".
            assert result.is_new_master is False

            # Close inbound streams to end the receivers (and run())
            em_in_tx.close()
            cm_tx.close()
            co_tx.close()

    # We should have updated seniority to 2 (A + B).
    assert election.seniority == 2


@pytest.mark.anyio
async def test_peer_with_higher_seniority_wins_and_we_switch_master() -> None:
    """
    If a peer with clearly higher seniority participates in the round, they should win.
    We should broadcast our status exactly once for this round, then switch master.
    """
    em_out_tx, em_out_rx = channel[ElectionMessage]()
    em_in_tx, em_in_rx = channel[ElectionMessage]()
    er_tx, er_rx = channel[ElectionResult]()
    cm_tx, cm_rx = channel[ConnectionMessage]()
    co_tx, co_rx = channel[ForwarderCommand]()

    election = Election(
        node_id=NodeId("ME"),
        election_message_receiver=em_in_rx,
        election_message_sender=em_out_tx,
        election_result_sender=er_tx,
        connection_message_receiver=cm_rx,
        command_receiver=co_rx,
        is_candidate=True,
    )

    async with create_task_group() as tg:
        with fail_after(2):
            tg.start_soon(election.run)

            # Start round with peer's message (higher seniority)
            await em_in_tx.send(em(clock=1, seniority=10, node_id="PEER"))

            # We should still broadcast our status exactly once for this round
            while True:
                got = await em_out_rx.receive()
                if got.clock == 1:
                    assert got.seniority == 0
                    break

            # After the timeout, election result for clock=1 should report the peer as master
            # (Skip any earlier result from the boot campaign at clock=0 by filtering on election_clock)
            while True:
                result = await er_rx.receive()
                if result.session_id.election_clock == 1:
                    break

            assert result.session_id.master_node_id == NodeId("PEER")
            assert result.is_new_master is True

            em_in_tx.close()
            cm_tx.close()
            co_tx.close()

    # We lost → seniority unchanged
    assert election.seniority == 0


@pytest.mark.anyio
async def test_duplicate_protocol_copies_count_as_one_candidate() -> None:
    """Legacy and isolated copies of one status cannot inflate an election."""

    em_out_tx, _em_out_rx = channel[ElectionMessage]()
    em_in_tx, em_in_rx = channel[ElectionMessage]()
    er_tx, _er_rx = channel[ElectionResult]()
    cm_tx, cm_rx = channel[ConnectionMessage]()
    co_tx, co_rx = channel[ForwarderCommand]()
    election = Election(
        node_id=NodeId("ME"),
        election_message_receiver=em_in_rx,
        election_message_sender=em_out_tx,
        election_result_sender=er_tx,
        connection_message_receiver=cm_rx,
        command_receiver=co_rx,
    )
    peer_status = em(clock=1, seniority=3, node_id="PEER")

    async with create_task_group() as task_group:
        with fail_after(2):
            task_group.start_soon(election.run)
            await em_in_tx.send(peer_status)
            await em_in_tx.send(peer_status)
            while peer_status not in election._candidates:  # pyright: ignore[reportPrivateUsage]
                await anyio.sleep(0.01)
            await anyio.sleep(0.02)
            assert election._candidates.count(peer_status) == 1  # pyright: ignore[reportPrivateUsage]

            em_in_tx.close()
            cm_tx.close()
            co_tx.close()


@pytest.mark.anyio
async def test_ignores_older_messages() -> None:
    """
    Messages with a lower clock than the current round are ignored by the receiver.
    Expect exactly one broadcast for the higher clock round.
    """
    em_out_tx, em_out_rx = channel[ElectionMessage]()
    em_in_tx, em_in_rx = channel[ElectionMessage]()
    er_tx, _er_rx = channel[ElectionResult]()
    cm_tx, cm_rx = channel[ConnectionMessage]()
    co_tx, co_rx = channel[ForwarderCommand]()

    election = Election(
        node_id=NodeId("ME"),
        election_message_receiver=em_in_rx,
        election_message_sender=em_out_tx,
        election_result_sender=er_tx,
        connection_message_receiver=cm_rx,
        command_receiver=co_rx,
        is_candidate=True,
    )

    async with create_task_group() as tg:
        with fail_after(2):
            tg.start_soon(election.run)

            # Newer round arrives first -> triggers campaign at clock=2
            await em_in_tx.send(em(clock=2, seniority=0, node_id="A"))
            while True:
                first = await em_out_rx.receive()
                if first.clock == 2:
                    break

            # Older message (clock=1) must be ignored (no second broadcast)
            await em_in_tx.send(em(clock=1, seniority=999, node_id="B"))

            got_second = False
            with move_on_after(0.05):
                _ = await em_out_rx.receive()
                got_second = True
            assert not got_second, "Should not receive a broadcast for an older round"

            em_in_tx.close()
            cm_tx.close()
            co_tx.close()

    # Not asserting on the result; focus is on ignore behavior.


@pytest.mark.anyio
async def test_two_rounds_emit_two_broadcasts_and_increment_clock() -> None:
    """
    Two successive rounds → two broadcasts. Second round triggered by a higher-clock message.
    """
    em_out_tx, em_out_rx = channel[ElectionMessage]()
    em_in_tx, em_in_rx = channel[ElectionMessage]()
    er_tx, _er_rx = channel[ElectionResult]()
    cm_tx, cm_rx = channel[ConnectionMessage]()
    co_tx, co_rx = channel[ForwarderCommand]()

    election = Election(
        node_id=NodeId("ME"),
        election_message_receiver=em_in_rx,
        election_message_sender=em_out_tx,
        election_result_sender=er_tx,
        connection_message_receiver=cm_rx,
        command_receiver=co_rx,
        is_candidate=True,
    )

    async with create_task_group() as tg:
        with fail_after(2):
            tg.start_soon(election.run)

            # Round 1 at clock=1
            await em_in_tx.send(em(clock=1, seniority=0, node_id="X"))
            while True:
                m1 = await em_out_rx.receive()
                if m1.clock == 1:
                    break

            # Round 2 at clock=2
            await em_in_tx.send(em(clock=2, seniority=0, node_id="Y"))
            while True:
                m2 = await em_out_rx.receive()
                if m2.clock == 2:
                    break

            em_in_tx.close()
            cm_tx.close()
            co_tx.close()

    # Not asserting on who won; just that both rounds were broadcast.


@pytest.mark.anyio
async def test_promotion_new_seniority_counts_participants() -> None:
    """
    When we win against two peers in the same round, our seniority becomes
    max(existing, number_of_candidates). With existing=0: expect 3 (us + A + B).
    """
    em_out_tx, em_out_rx = channel[ElectionMessage]()
    em_in_tx, em_in_rx = channel[ElectionMessage]()
    er_tx, er_rx = channel[ElectionResult]()
    cm_tx, cm_rx = channel[ConnectionMessage]()
    co_tx, co_rx = channel[ForwarderCommand]()

    election = Election(
        node_id=NodeId("ME"),
        election_message_receiver=em_in_rx,
        election_message_sender=em_out_tx,
        election_result_sender=er_tx,
        connection_message_receiver=cm_rx,
        command_receiver=co_rx,
        is_candidate=True,
    )

    async with create_task_group() as tg:
        with fail_after(2):
            tg.start_soon(election.run)

            # Start round at clock=7 with two peer participants
            await em_in_tx.send(em(clock=7, seniority=0, node_id="A"))
            await em_in_tx.send(em(clock=7, seniority=0, node_id="B"))

            # We should see exactly one broadcast from us for this round
            while True:
                got = await em_out_rx.receive()
                if got.clock == 7 and got.proposed_session.master_node_id == NodeId(
                    "ME"
                ):
                    break

            # Wait for the election to finish so seniority updates
            _ = await er_rx.receive()

            em_in_tx.close()
            cm_tx.close()
            co_tx.close()

    # We + A + B = 3 → new seniority expected to be 3
    assert election.seniority == 3


@pytest.mark.anyio
async def test_connection_message_triggers_new_round_broadcast() -> None:
    """
    A connection message increments the clock and starts a new campaign.
    We should observe a broadcast at the incremented clock.
    """
    em_out_tx, em_out_rx = channel[ElectionMessage]()
    em_in_tx, em_in_rx = channel[ElectionMessage]()
    er_tx, _er_rx = channel[ElectionResult]()
    cm_tx, cm_rx = channel[ConnectionMessage]()
    co_tx, co_rx = channel[ForwarderCommand]()

    election = Election(
        node_id=NodeId("ME"),
        election_message_receiver=em_in_rx,
        election_message_sender=em_out_tx,
        election_result_sender=er_tx,
        connection_message_receiver=cm_rx,
        command_receiver=co_rx,
        is_candidate=True,
    )

    async with create_task_group() as tg:
        with fail_after(2):
            tg.start_soon(election.run)

            # Send any connection message object; we close quickly to cancel before result creation
            await cm_tx.send(ConnectionMessage(node_id=NodeId(), connected=True))

            # Expect a broadcast for the new round at clock=1
            while True:
                got = await em_out_rx.receive()
                if got.clock == 1 and got.proposed_session.master_node_id == NodeId(
                    "ME"
                ):
                    break

            # Close promptly to avoid waiting for campaign completion
            em_in_tx.close()
            cm_tx.close()
            co_tx.close()

    # After cancellation (before election finishes), no seniority changes asserted here.


@pytest.mark.anyio
async def test_tie_breaker_prefers_node_with_more_commands_seen() -> None:
    """
    With equal seniority, the node that has seen more commands should win the election.
    We increase our local 'commands_seen' by sending TestCommand()s before triggering the round.
    """
    em_out_tx, em_out_rx = channel[ElectionMessage]()
    em_in_tx, em_in_rx = channel[ElectionMessage]()
    er_tx, er_rx = channel[ElectionResult]()
    cm_tx, cm_rx = channel[ConnectionMessage]()
    co_tx, co_rx = channel[ForwarderCommand]()

    me = NodeId("ME")

    election = Election(
        node_id=me,
        election_message_receiver=em_in_rx,
        election_message_sender=em_out_tx,
        election_result_sender=er_tx,
        connection_message_receiver=cm_rx,
        command_receiver=co_rx,
        is_candidate=True,
        seniority=0,
    )

    async with create_task_group() as tg:
        with fail_after(2):
            tg.start_soon(election.run)

            # Pump local commands so our commands_seen is high before the round starts
            for _ in range(50):
                await co_tx.send(
                    ForwarderCommand(origin=SystemId("SOMEONE"), command=TestCommand())
                )

            # Trigger a round at clock=1 with a peer of equal seniority but fewer commands
            await em_in_tx.send(
                em(clock=1, seniority=0, node_id="PEER", commands_seen=5)
            )

            # Observe our broadcast for this round (to ensure we've joined the round)
            while True:
                got = await em_out_rx.receive()
                if got.clock == 1 and got.proposed_session.master_node_id == me:
                    # We don't assert exact count, just that we've participated this round.
                    break

            # The elected result for clock=1 should be us due to higher commands_seen
            while True:
                result = await er_rx.receive()
                if result.session_id.master_node_id == me:
                    assert result.session_id.election_clock in (0, 1)
                    break

            em_in_tx.close()
            cm_tx.close()
            co_tx.close()


# ===================================================== #
#   Connection-churn / livelock regression (#400)       #
# ===================================================== #


def _make_election(node_id: str) -> Election:
    """Build an Election wired to fresh channels for unit-level assertions."""
    em_out_tx, _em_out_rx = channel[ElectionMessage]()
    _em_in_tx, em_in_rx = channel[ElectionMessage]()
    er_tx, _er_rx = channel[ElectionResult]()
    _cm_tx, cm_rx = channel[ConnectionMessage]()
    _co_tx, co_rx = channel[ForwarderCommand]()
    return Election(
        node_id=NodeId(node_id),
        election_message_receiver=em_in_rx,
        election_message_sender=em_out_tx,
        election_result_sender=er_tx,
        connection_message_receiver=cm_rx,
        command_receiver=co_rx,
        is_candidate=True,
    )


def test_apply_connection_updates_reports_only_real_membership_changes() -> None:
    """The membership diff drives whether a connection update starts a round.

    Only a peer genuinely joining or leaving counts; duplicate connects,
    disconnects of untracked peers, and updates about ourselves are no-ops.
    """
    election = _make_election("ME")

    # First sighting of a peer (its first connection) is a change.
    assert election._apply_connection_updates(  # pyright: ignore[reportPrivateUsage]
        [ConnectionMessage(node_id=NodeId("A"), connected=True)]
    )
    # Its single connection closing (last close) is a change: the peer leaves.
    assert election._apply_connection_updates(  # pyright: ignore[reportPrivateUsage]
        [ConnectionMessage(node_id=NodeId("A"), connected=False)]
    )
    # A disconnect for an untracked peer is not a change.
    assert not election._apply_connection_updates(  # pyright: ignore[reportPrivateUsage]
        [ConnectionMessage(node_id=NodeId("A"), connected=False)]
    )
    # Updates about ourselves never count.
    assert not election._apply_connection_updates(  # pyright: ignore[reportPrivateUsage]
        [ConnectionMessage(node_id=NodeId("ME"), connected=True)]
    )
    # A burst that nets out to a single new peer counts once.
    assert election._apply_connection_updates(  # pyright: ignore[reportPrivateUsage]
        [
            ConnectionMessage(node_id=NodeId("B"), connected=True),
            ConnectionMessage(node_id=NodeId("B"), connected=True),
        ]
    )


def test_apply_connection_updates_refcounts_multihomed_peer() -> None:
    """A multi-homed peer stays present until its LAST connection closes.

    The transport emits one update per connection, so dropping the peer on the
    first close (and then ignoring the real last close as an untracked-peer
    disconnect) could miss a genuine master loss. Ref-counting prevents that.
    """
    election = _make_election("ME")

    # Two connections to the same peer: the first is a join, the second is not.
    assert election._apply_connection_updates(  # pyright: ignore[reportPrivateUsage]
        [ConnectionMessage(node_id=NodeId("A"), connected=True)]
    )
    assert not election._apply_connection_updates(  # pyright: ignore[reportPrivateUsage]
        [ConnectionMessage(node_id=NodeId("A"), connected=True)]
    )
    # First of the two connections closes: peer is still present -> no change.
    assert not election._apply_connection_updates(  # pyright: ignore[reportPrivateUsage]
        [ConnectionMessage(node_id=NodeId("A"), connected=False)]
    )
    # Last connection closes: the peer genuinely leaves -> change.
    assert election._apply_connection_updates(  # pyright: ignore[reportPrivateUsage]
        [ConnectionMessage(node_id=NodeId("A"), connected=False)]
    )
    # And an extra close past zero is a harmless no-op (no negative counts).
    assert not election._apply_connection_updates(  # pyright: ignore[reportPrivateUsage]
        [ConnectionMessage(node_id=NodeId("A"), connected=False)]
    )


@pytest.mark.anyio
async def test_duplicate_connection_update_does_not_restart_round(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """A redundant connect to an already-tracked peer must not start a round.

    This is the core of the livelock fix: pre-fix every connection message
    bumped the clock and restarted the campaign; now only membership changes do.
    """
    monkeypatch.setattr("skulk.shared.election.CONNECTION_SETTLE_DELAY", 0.02)

    em_out_tx, _em_out_rx = channel[ElectionMessage]()
    em_in_tx, em_in_rx = channel[ElectionMessage]()
    er_tx, _er_rx = channel[ElectionResult]()
    cm_tx, cm_rx = channel[ConnectionMessage]()
    co_tx, co_rx = channel[ForwarderCommand]()

    election = Election(
        node_id=NodeId("ME"),
        election_message_receiver=em_in_rx,
        election_message_sender=em_out_tx,
        election_result_sender=er_tx,
        connection_message_receiver=cm_rx,
        command_receiver=co_rx,
        is_candidate=True,
    )

    async with create_task_group() as tg:
        with fail_after(2):
            tg.start_soon(election.run)

            # First sighting of peer A: a real membership change -> clock 1.
            await cm_tx.send(ConnectionMessage(node_id=NodeId("A"), connected=True))
            while election.clock != 1:
                await anyio.sleep(0.01)

            # A duplicate connect to A must NOT advance the clock / start a round.
            await cm_tx.send(ConnectionMessage(node_id=NodeId("A"), connected=True))
            await anyio.sleep(0.2)
            assert election.clock == 1

        em_in_tx.close()
        cm_tx.close()
        co_tx.close()
        await election.shutdown()


@pytest.mark.anyio
async def test_connection_churn_burst_still_converges(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """A burst of redundant connection updates must not starve convergence.

    Pre-fix, updates arriving faster than DEFAULT_ELECTION_TIMEOUT cancelled and
    restarted the campaign forever, so no ElectionResult was ever produced. The
    membership-diff gate collapses the burst to a single round that completes.
    """
    monkeypatch.setattr("skulk.shared.election.CONNECTION_SETTLE_DELAY", 0.02)

    em_out_tx, _em_out_rx = channel[ElectionMessage]()
    _em_in_tx, em_in_rx = channel[ElectionMessage]()
    er_tx, er_rx = channel[ElectionResult]()
    cm_tx, cm_rx = channel[ConnectionMessage]()
    _co_tx, co_rx = channel[ForwarderCommand]()

    election = Election(
        node_id=NodeId("ME"),
        election_message_receiver=em_in_rx,
        election_message_sender=em_out_tx,
        election_result_sender=er_tx,
        connection_message_receiver=cm_rx,
        command_receiver=co_rx,
        is_candidate=True,
    )

    async def churn() -> None:
        # Hammer the same peer faster than the (fast) election timeout.
        for _ in range(40):
            await cm_tx.send(ConnectionMessage(node_id=NodeId("A"), connected=True))
            await anyio.sleep(0.01)

    async with create_task_group() as tg:
        with fail_after(3):
            tg.start_soon(election.run)
            tg.start_soon(churn)

            # Despite the churn, the election must still produce a result for the
            # round that admitted peer A (clock >= 1).
            while True:
                result = await er_rx.receive()
                if result.session_id.election_clock >= 1 or result.won_clock >= 1:
                    break

        # Tear down cleanly while churn may still be sending: cancel rather than
        # close the channel out from under the running producer task.
        tg.cancel_scope.cancel()


@pytest.mark.anyio
async def test_late_same_round_candidate_corrects_completed_result() -> None:
    """A peer's delayed vote must converge after our local campaign timeout.

    With asymmetric subscription readiness, one peer can finish its round
    before another peer even receives the initiating broadcast. Both still
    exchange messages on the same live connection, so no membership change
    comes along to repair their conflicting completed results.
    """
    outbound, _outbound_receiver = channel[ElectionMessage]()
    inbound, inbound_receiver = channel[ElectionMessage]()
    results, result_receiver = channel[ElectionResult]()
    _connections, connection_receiver = channel[ConnectionMessage]()
    _commands, command_receiver = channel[ForwarderCommand]()
    election = Election(
        node_id=NodeId("B"),
        election_message_receiver=inbound_receiver,
        election_message_sender=outbound,
        election_result_sender=results,
        connection_message_receiver=connection_receiver,
        command_receiver=command_receiver,
    )
    async with create_task_group() as tasks:
        tasks.start_soon(election.run)
        with fail_after(2):
            initial = await result_receiver.receive()
            assert initial.won_clock == 0
            assert initial.session_id.master_node_id == NodeId("B")
            # The candidate's original vote has equal seniority; our post-win
            # seniority increment must not rewrite the already completed vote.
            late = em(clock=0, seniority=0, node_id="Z")
            await inbound.send(late)
            corrected = await result_receiver.receive()
            assert corrected.won_clock == 0
            assert corrected.session_id == late.proposed_session
            assert corrected.is_new_master
            # Duplicate wire copies and losing proposals cannot emit another
            # correction or move the cluster back to an inferior result.
            await inbound.send(late)
            await inbound.send(em(clock=0, seniority=0, node_id="A"))
            with move_on_after(0.05):
                unexpected = await result_receiver.receive()
                pytest.fail(f"unexpected repeated election result: {unexpected}")
        tasks.cancel_scope.cancel()
