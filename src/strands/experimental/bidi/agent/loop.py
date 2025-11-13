"""Agent loop.

The agent loop handles the events received from the model and executes tools when given a tool use request.
"""

import logging
from typing import AsyncIterable, TYPE_CHECKING

from ..types.events import BidiOutputEvent, BidiTranscriptStreamEvent
from ....types._events import ToolResultEvent, ToolResultMessageEvent, ToolStreamEvent, ToolUseStreamEvent
from ....types.content import Message
from ....types.tools import ToolResult, ToolUse

if TYPE_CHECKING:
    from .agent import BidiAgent

logger = logging.getLogger(__name__)


class _BidiAgentLoop:
    """Agent loop."""

    def __init__(self, agent: "BidiAgent") -> None:
        """Initialize members of the agent loop.

        Note, before receiving events from the loop, the user must call `start`.

        Args:
            agent: Bidirectional agent to loop over.
        """
        self._agent = agent
        self._active = False  # flag if agent loop is started

    async def start(self) -> None:
        """Start the agent loop.
        
        The agent model is started as part of this call.
        """
        if self.active:
            return

        logger.debug("starting agent loop")

        await self._agent.model.start(
            system_prompt=self._agent.system_prompt,
            tools=self._agent.tool_registry.get_all_tool_specs(),
            messages=self._agent.messages,
        )

        self._active = True

    async def stop(self) -> None:
        """Stop the agent loop."""
        if not self.active:
            return

        logger.debug("stopping agent loop")

        await self._agent.model.stop()
        self._active = False

    async def receive(self) -> AsyncIterable[BidiOutputEvent]:
        """Receive model and tool call events."""
        logger.debug("running model")

        async for event in self._agent.model.receive():
            if not self.active:
                break

            yield event

            if isinstance(event, BidiTranscriptStreamEvent):
                if event["is_final"]:
                    message: Message = {"role": event["role"], "content": [{"text": event["text"]}]}
                    self._agent.messages.append(message)

            elif isinstance(event, ToolUseStreamEvent):
                async for tool_event in self._run_tool(event["current_tool_use"]):
                    yield tool_event

    @property
    def active(self) -> bool:
        """True if agent loop started, False otherwise."""
        return self._active

    async def _run_tool(self, tool_use: ToolUse) -> AsyncIterable[BidiOutputEvent]:
        """Task for running tool requested by the model."""
        logger.debug("running tool")

        result: ToolResult = None

        try:
            tool = self._agent.tool_registry.registry[tool_use["name"]]
            invocation_state = {}

            async for event in tool.stream(tool_use, invocation_state):
                if isinstance(event, ToolResultEvent):
                    yield event
                    result = event.tool_result
                    break

                if isinstance(event, ToolStreamEvent):
                    yield event
                else:
                    yield ToolStreamEvent(tool_use, event)

        except Exception as e:
            result = {
                "toolUseId": tool_use["toolUseId"],
                "status": "error",
                "content": [{"text": f"Error: {str(e)}"}]
            }

        await self._agent.model.send(ToolResultEvent(result))

        message: Message = {
            "role": "user",
            "content": [{"toolResult": result}],
        }
        self._agent.messages.append(message)
        yield ToolResultMessageEvent(message)
