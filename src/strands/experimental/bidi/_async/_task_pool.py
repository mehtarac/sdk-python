"""<TODO>."""
import asyncio
from typing import Awaitable

class _TaskPool:
    """<TODO>."""

    def __init__(self) -> None:
        """<TODO>."""
        self._tasks = set()

    def create_task(self, coro: Awaitable[None]) -> None:
        """Utilitly to create async task.

        Adds a clean up callback to run after task completes.
        """
        task = asyncio.create_task(coro)
        task.add_done_callback(lambda task: self._tasks.remove(task))

        self._tasks.add(task)

    async def cancel(self) -> None:
        """<TODO>."""
        for task in self._tasks:
            task.cancel()
        
        await asyncio.gather(*self._tasks)
