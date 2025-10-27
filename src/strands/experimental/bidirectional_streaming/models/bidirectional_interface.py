"""Unified bidirectional streaming interface.

Single layer combining model and session abstractions for simpler implementation.
"""

from typing import Any, AsyncIterable, Protocol, Union

from ....types.content import Messages
from ....types.tools import ToolSpec, ToolResult
from ..types.bidirectional_streaming import (
    AudioInputEvent,
    BidirectionalStreamEvent,
    ImageInputEvent,
)


class BidirectionalInterface(Protocol):
    """Unified interface for bidirectional streaming models.
    
    Combines model configuration and session communication in a single abstraction.
    Providers implement this directly without separate model/session classes.
    """

    def get_config(self) -> dict[str, Any]:
        """Get current configuration."""
        ...

    def update_config(self, **model_config: Any) -> None:
        """Update model configuration."""
        ...

    async def start(
        self,
        system_prompt: str | None = None,
        tools: list[ToolSpec] | None = None,
        messages: Messages | None = None,
        **kwargs,
    ) -> None:
        """Establish bidirectional connection with the model."""
        ...

    async def close(self) -> None:
        """Close connection and cleanup resources."""
        ...

    # Communication
    async def receive(self) -> AsyncIterable[BidirectionalStreamEvent]:
        """Receive events from the model in standardized format."""
        ...

    async def send(self, content: Union[str, ImageInputEvent, AudioInputEvent, ToolResult]) -> None:
        """Send structured content to the model."""
        ...

    # TODO: Discuss if we need to expose this to users
    async def send_interrupt(self) -> None:
        """Send interruption signal to stop generation."""
        ...