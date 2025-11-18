"""Agent loop.

The agent loop handles the events received from the model and executes tools when given a tool use request.
"""

import asyncio
import logging
from typing import TYPE_CHECKING, Any, AsyncIterable

from .._async import _TaskPool, start, stop
from ..hooks.events import (
    BidiAfterInvocationEvent,
    BidiAfterToolCallEvent,
    BidiBeforeInvocationEvent,
    BidiBeforeToolCallEvent,
    BidiInterruptionEvent as BidiInterruptionHookEvent,
    BidiMessageAddedEvent,
)
from ..types.events import BidiInterruptionEvent, BidiOutputEvent, BidiTranscriptStreamEvent
from ....types._events import ToolResultEvent, ToolResultMessageEvent, ToolStreamEvent, ToolUseStreamEvent
from ....types.content import Message
from ....types.tools import ToolResult, ToolUse

if TYPE_CHECKING:
    from .agent import BidiAgent

logger = logging.getLogger(__name__)


class _BidiAgentLoop:
    """Agent loop.

    Attributes:
        _agent: BidiAgent instance to loop.
        _started: Flag if agent loop has started.
        _event_queue: Queue model and tool call events for receiver.
        _task_pool: Track active async tasks created in loop.
    """

    _event_queue: asyncio.Queue
    _task_pool: _TaskPool

    def __init__(self, agent: "BidiAgent") -> None:
        """Initialize members of the agent loop.

        Note, before receiving events from the loop, the user must call `start`.

        Args:
            agent: Bidirectional agent to loop over.
        """
        self._agent = agent
        self._started = False

    @start
    async def start(self) -> None:
        """Start the agent loop.

        The agent model is started as part of this call.
        """
        if self._started:
            raise RuntimeError("call stop before starting again")

        logger.debug("agent loop starting")

        await self._agent.hooks.invoke_callbacks_async(BidiBeforeInvocationEvent(agent=self._agent))

        await self._agent.model.start(
            system_prompt=self._agent.system_prompt,
            tools=self._agent.tool_registry.get_all_tool_specs(),
            messages=self._agent.messages,
        )

        self._event_queue = asyncio.Queue(maxsize=1)

        self._task_pool = _TaskPool()
        self._task_pool.create_task(self._run_model())

        self._started = True

    async def stop(self) -> None:
        """Stop the agent loop."""

        logger.debug("agent loop stopping")
        self._started = False

        async def stop_tasks() -> None:
            await self.task_pool.cancel()

        async def stop_event_queue() -> None:
            self._event_queue.shutdown(immediate=True)

        async def stop_model() -> None:
            await self._agent.model.stop()

        try:
            await stop(stop_tasks, stop_event_queue, stop_model)
        finally:
            await self._agent.hooks.invoke_callbacks_async(BidiAfterInvocationEvent(agent=self._agent))

    async def receive(self) -> AsyncIterable[BidiOutputEvent]:
        """Receive model and tool call events."""
        if not self._started:
            raise RuntimeError("must call start")

        while True:
            event = await self._event_queue.get()
            yield event

    async def _run_model(self) -> None:
        """Task for running the model.

        Events are streamed through the event queue.
        """
        logger.debug("model task starting")

        async for event in self._agent.model.receive():
            await self._event_queue.put(event)

            if isinstance(event, BidiTranscriptStreamEvent):
                if event["is_final"]:
                    message: Message = {"role": "assistant", "content": [{"text": event["text"]}]}
                    self._agent.messages.append(message)
                    await self._agent.hooks.invoke_callbacks_async(
                        BidiMessageAddedEvent(agent=self._agent, message=message)
                    )

            elif isinstance(event, ToolUseStreamEvent):
                tool_use = event["current_tool_use"]
                self._create_task(self._run_tool(tool_use))

                tool_message: Message = {"role": "assistant", "content": [{"toolUse": tool_use}]}
                self._agent.messages.append(tool_message)
                await self._agent.hooks.invoke_callbacks_async(
                    BidiMessageAddedEvent(agent=self._agent, message=message)
                )

            elif isinstance(event, BidiInterruptionEvent):
                await self._agent.hooks.invoke_callbacks_async(
                    BidiInterruptionHookEvent(
                        agent=self._agent,
                        reason=event["reason"],
                        interrupted_response_id=event.get("interrupted_response_id"),
                    )
                )

    async def _run_tool(self, tool_use: ToolUse) -> None:
        """Task for running tool requested by the model."""
        logger.debug("tool_name=<%s> | tool execution starting", tool_use["name"])

        result: ToolResult
        exception: Exception | None = None
        tool = None
        invocation_state: dict[str, Any] = {}

        try:
            tool = self._agent.tool_registry.registry[tool_use["name"]]

            await self._agent.hooks.invoke_callbacks_async(
                BidiBeforeToolCallEvent(
                    agent=self._agent,
                    selected_tool=tool,
                    tool_use=tool_use,
                    invocation_state=invocation_state,
                )
            )

            async for event in tool.stream(tool_use, invocation_state):
                if isinstance(event, ToolResultEvent):
                    await self._event_queue.put(event)
                    result = event.tool_result
                    break

                if isinstance(event, ToolStreamEvent):
                    await self._event_queue.put(event)
                else:
                    await self._event_queue.put(ToolStreamEvent(tool_use, event))

        except Exception as e:
            result = {"toolUseId": tool_use["toolUseId"], "status": "error", "content": [{"text": f"Error: {str(e)}"}]}

        finally:
            await self._agent.hooks.invoke_callbacks_async(
                BidiAfterToolCallEvent(
                    agent=self._agent,
                    selected_tool=tool,
                    tool_use=tool_use,
                    invocation_state=invocation_state,
                    result=result,
                    exception=exception,
                )
            )

        await self._agent.model.send(ToolResultEvent(result))

        message: Message = {
            "role": "user",
            "content": [{"toolResult": result}],
        }
        self._agent.messages.append(message)
        await self._agent.hooks.invoke_callbacks_async(BidiMessageAddedEvent(agent=self._agent, message=message))
        await self._event_queue.put(ToolResultMessageEvent(message))
