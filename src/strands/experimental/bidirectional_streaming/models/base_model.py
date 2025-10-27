""" " Abstract interface for bidirectional streaming models."""

import abc
from typing import Any, Protocol

from ....types.content import Messages
from ....types.tools import ToolSpec
from .base_session import BidirectionalModelSession


class BidirectionalModel(Protocol):
    """Abstract interface for bidirectional streaming models.

    Manages configuration and creates stateful sessions.
    """


    def get_config(self) -> dict[str, Any]:
        """Get current configuration."""
        pass


    def update_config(self, **model_config: Any) -> None:
        """Update model configuration."""
        pass


    async def create_bidirectional_connection(
        self,
        system_prompt: str | None = None,
        tools: list[ToolSpec] | None = None,
        messages: Messages | None = None,
        **kwargs,
    ) -> BidirectionalModelSession:
        """Create a new stateful bidirectional session.

        Model (this) manages configuration and client.
        Session (returned) manages connection state and communication.
        """
        raise NotImplementedError
