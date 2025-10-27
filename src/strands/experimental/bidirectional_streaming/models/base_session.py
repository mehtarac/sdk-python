"""Bidirectional model interface for real-time streaming conversations.

Defines the interface for models that support bidirectional streaming capabilities.
Provides abstractions for different model providers with connection-based communication
patterns that support real-time audio and text interaction.

Features:
- connection-based persistent connections
- Real-time bidirectional communication
- Provider-agnostic event normalization
- Tool execution integration
"""


import logging
from typing import AsyncIterable, Union,Protocol

from ....types.tools import ToolResult
from ..types.bidirectional_streaming import (
    AudioInputEvent,
    BidirectionalStreamEvent,
    ImageInputEvent,
)

logger = logging.getLogger(__name__)


class BidirectionalModelSession(Protocol):
    """Abstract interface for model-specific bidirectional communication connections.

    Defines the contract for managing persistent streaming connections with individual
    model providers, handling audio/text input, receiving events, and managing
    tool execution results.
    """


    async def receive(self) -> AsyncIterable[BidirectionalStreamEvent]:
        """Receive events from the model in standardized format.

        Converts provider-specific events to a common format that can be
        processed uniformly by the event loop.
        """
        raise NotImplementedError


    async def send(self, content: Union[str, ImageInputEvent, AudioInputEvent, ToolResult]) -> None:
        """Send structured content (text, images,audio tool results) to the model.

        Args:
            content: Text string, ImageInputEvent, AudioInputEvent or ToolResult
        """
        raise NotImplementedError


    # TODO: discuss if we need this or not?
    async def send_interrupt(self) -> None:
        """Send interruption signal to stop generation immediately.

        Enables responsive conversational experiences where users can
        naturally interrupt during model responses.
        """
        raise NotImplementedError

    async def close(self) -> None:
        """Close the connection and cleanup resources."""
        raise NotImplementedError
