"""The CLI must not destroy loop-bound resources between creation and serving."""

import asyncio

import pytest

from skulk.main import Args, Node, run_node


async def test_create_and_run_share_a_live_event_loop(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """A task started during construction survives into run and can be reaped."""
    loop = asyncio.get_running_loop()
    started = asyncio.Event()
    background: asyncio.Task[None] | None = None

    async def work() -> None:
        started.set()
        await asyncio.Event().wait()

    async def create(cls: type[Node], args: Args) -> Node:
        nonlocal background
        assert asyncio.get_running_loop() is loop
        background = asyncio.create_task(work())
        return object.__new__(Node)

    async def run(self: Node) -> None:
        assert asyncio.get_running_loop() is loop
        await started.wait()
        assert background is not None and not background.done()
        background.cancel()
        with pytest.raises(asyncio.CancelledError):
            await background

    monkeypatch.setattr(Node, "create", classmethod(create))
    monkeypatch.setattr(Node, "run", run)
    await run_node(Args(libp2p_port=0))
