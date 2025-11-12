"""Handle websocket input and output from bidi agent."""

import asyncio
import logging
from typing import cast

from fastapi import WebSocket

from ..types.io import BidiInput, BidiOutput
from ..types.events import BidiInputEvent, BidiOutputEvent

logger = logging.getLogger(__name__)


class _BidiWebSocketInput(BidiInput):
    """Handle input from websocket for bidi agent."""

    def __init__(self, websocket: WebSocket) -> None:
        """Set websocket instance."""
        self._websocket = websocket

    async def __call__(self) -> BidiInputEvent:
        """Translate input from websocket into bidi agent input."""
        return cast(BidiInputEvent, await self._websocket.receive_json())


class _BidiWebSocketOutput(BidiOutput):
    """Handle output from bidi agent for websocket."""

    def __init__(self, websocket: WebSocket, event_types: list[str] | None = None) -> None:
        """Set websocket instance."""
        self._websocket = websocket
        self._event_types = None if event_types is None else set(event_types)

    async def __call__(self, event: BidiOutputEvent) -> None:
        """Translate output from bidi agent into websocket output."""
        await self._websocket.send_json(event.as_dict())

        # TODO:
        await asyncio.sleep(0.02)

class BidiWebSocketIO:
    """Handle text input and output from bidi agent."""

    def __init__(self, websocket: WebSocket) -> None:
        """Set websocket instance."""
        self._websocket = websocket

    def input(self) -> _BidiWebSocketInput:
        """Return websocket BidiInput processor"""
        return _BidiWebSocketInput(self._websocket)

    def output(self) -> _BidiWebSocketOutput:
        """Return websocket BidiOutput processor"""
        return _BidiWebSocketOutput(self._websocket)
