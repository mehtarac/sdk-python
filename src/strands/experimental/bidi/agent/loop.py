"""Agent loop.

The agent loop handles the events received from the model and executes tools when given a tool use request.
"""

import asyncio
import logging
from typing import TYPE_CHECKING, Any, AsyncIterable, Awaitable

from opentelemetry import trace as trace_api

from ....telemetry.tracer import serialize
from ....types._events import ToolResultEvent, ToolResultMessageEvent, ToolStreamEvent, ToolUseStreamEvent
from ....types.content import Message
from ....types.tools import ToolResult, ToolUse
from ..hooks.events import (
    BidiAfterInvocationEvent,
    BidiAfterToolCallEvent,
    BidiBeforeInvocationEvent,
    BidiBeforeToolCallEvent,
    BidiMessageAddedEvent,
)
from ..hooks.events import (
    BidiInterruptionEvent as BidiInterruptionHookEvent,
)
from ..types.events import BidiInterruptionEvent, BidiOutputEvent, BidiTranscriptStreamEvent

if TYPE_CHECKING:
    from .agent import BidiAgent

logger = logging.getLogger(__name__)


class _BidiAgentLoop:
    """Agent loop.

    Attributes:
        _agent: BidiAgent instance to loop.
        _event_queue: Queue model and tool call events for receiver.
        _stop_event: Sentinel to mark end of loop.
        _tasks: Track active async tasks created in loop.
        _active: Flag if agent loop is started.
    """

    _event_queue: asyncio.Queue
    _stop_event: object
    _tasks: set
    _active: bool

    def __init__(self, agent: "BidiAgent") -> None:
        """Initialize members of the agent loop.

        Note, before receiving events from the loop, the user must call `start`.

        Args:
            agent: Bidirectional agent to loop over.
        """
        self._agent = agent
        self._active: bool = False
        self._active = False
        self._loop_span = None

    async def start(self) -> None:
        """Start the agent loop.

        The agent model is started as part of this call.
        """
        if self.active:
            return

        logger.debug("agent loop starting")

        self._event_queue = asyncio.Queue(maxsize=1)
        self._stop_event = object()
        self._tasks = set()

        # Emit before invocation event
        await self._agent.hooks.invoke_callbacks_async(BidiBeforeInvocationEvent(agent=self._agent))

        await self._agent.model.start(
            system_prompt=self._agent.system_prompt,
            tools=self._agent.tool_registry.get_all_tool_specs(),
            messages=self._agent.messages,
        )

        # Wrap the model loop in the session span context
        self._loop_span = self._agent.tracer._start_span("agent_loop", self._agent._session_span)
        with trace_api.use_span(self._loop_span):
            self._create_task(self._run_model())

        self._active = True

    async def stop(self) -> None:
        """Stop the agent loop."""
        if not self.active:
            return

        logger.debug("agent loop stopping")

        try:
            # Cancel all tasks
            for task in self._tasks:
                task.cancel()

            # Wait briefly for tasks to finish their current operations
            await asyncio.gather(*self._tasks, return_exceptions=True)

            # Stop the model
            await self._agent.model.stop()

            # Clean up the event queue
            if not self._event_queue.empty():
                self._event_queue.get_nowait()
            self._event_queue.put_nowait(self._stop_event)

            self._active = False

        finally:
            # Emit after invocation event (reverse order for cleanup)
            await self._agent.hooks.invoke_callbacks_async(BidiAfterInvocationEvent(agent=self._agent))

    async def receive(self) -> AsyncIterable[BidiOutputEvent]:
        """Receive model and tool call events."""
        while True:
            event = await self._event_queue.get()
            if event is self._stop_event:
                break

            yield event

    @property
    def active(self) -> bool:
        """True if agent loop started, False otherwise."""
        return self._active

    def _create_task(self, coro: Awaitable[None]) -> None:
        """Utility to create async task.

        Adds a cleanup callback to run after task completes.
        """
        task: asyncio.Task[None] = asyncio.create_task(coro)  # type: ignore
        task.add_done_callback(lambda task: self._tasks.remove(task))

        self._tasks.add(task)

    async def _run_model(self) -> None:
        """Task for running the model.

        Events are streamed through the event queue.
        """
        logger.debug("model task starting")

        async for event in self._agent.model.receive():  # type: ignore
            await self._event_queue.put(event)

            if isinstance(event, BidiTranscriptStreamEvent):
                self._agent.tracer.handle_transcript_event(event, self._agent, self._loop_span)
                if event["is_final"]:
                    message: Message = {"role": event["role"], "content": [{"text": event["text"]}]}
                    await self._agent.hooks.invoke_callbacks_async(
                        BidiMessageAddedEvent(agent=self._agent, message=message)
                    )

            elif isinstance(event, ToolUseStreamEvent):
                tool_use = self._agent.tracer.handle_tool_use_event(event, self._agent, self._loop_span)
                self._create_task(self._run_tool(tool_use))

            elif isinstance(event, BidiInterruptionEvent):
                # Emit interruption hook event
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

        tool_span = self._agent.tracer.start_tool_call_span(tool=tool_use, parent_span=self._agent.tracer._model_span)

        tool_span.set_attribute(
            "gen_ai.input.messages", serialize([{"role": "assistant", "content": [{"toolUse": tool_use}]}])
        )

        try:
            tool = self._agent.tool_registry.registry[tool_use["name"]]

            # Emit before tool call event
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

                tool_span.set_attribute(
                    "gen_ai.output.messages", serialize([{"role": "user", "content": [{"toolResult": result}]}])
                )
            self._agent.tracer.end_tool_call_span(tool_span, result)

        except Exception as e:
            result = {"toolUseId": tool_use["toolUseId"], "status": "error", "content": [{"text": f"Error: {str(e)}"}]}
            # End tool span with error
            if result:
                tool_span.set_attribute(
                    "gen_ai.output.messages", serialize([{"role": "user", "content": [{"toolResult": result}]}])
                )
            self._agent.tracer.end_tool_call_span(tool_span, result, error=e)

        finally:
            # Emit after tool call event (reverse order for cleanup)
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
