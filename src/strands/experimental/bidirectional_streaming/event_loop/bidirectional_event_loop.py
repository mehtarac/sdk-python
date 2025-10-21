"""Class-based event loop for real-time bidirectional streaming with concurrent tool execution."""

import asyncio
import logging
import traceback
import uuid
from typing import TYPE_CHECKING, Optional, Dict, List

from ....tools._validator import validate_and_prepare_tools
from ....telemetry.metrics import Trace
from ....types._events import ToolResultEvent, ToolStreamEvent
from ....types.content import Message
from ....types.tools import ToolResult, ToolUse
from ....hooks import MessageAddedEvent
from ..models.bidirectional_model import BidirectionalModelSession

if TYPE_CHECKING:
    from ..agent import BidirectionalAgent

logger = logging.getLogger(__name__)


class BidirectionalEventLoop:
    """Event loop coordinator for bidirectional streaming sessions.
    
    Manages concurrent background tasks for model event processing and session supervision.
    Tool execution uses immediate asyncio.Task creation (0ms scheduling) rather than polling.
    Provides atomic interruption handling and race condition prevention.
    """

    def __init__(self, model_session: BidirectionalModelSession, agent: "BidirectionalAgent"):
        """Initialize event loop with model session and agent dependencies."""
        self.model_session = model_session
        self.agent = agent
        self.active = True
        
        # Task tracking
        self.background_tasks: List[asyncio.Task] = []
        self.pending_tool_tasks: Dict[str, asyncio.Task] = {}
        
        # Synchronization primitives
        self.interrupted = False
        self.interruption_lock = asyncio.Lock()
        self.conversation_lock = asyncio.Lock()  # Race condition prevention
        
        # Audio and metrics
        self.audio_output_queue = asyncio.Queue()
        self.tool_count = 0
        
        logger.debug("BidirectionalEventLoop initialized")

    async def start(self) -> None:
        """Start background tasks for model event processing and session supervision."""
        logger.debug("Starting bidirectional event loop")
        
        self.background_tasks = [
            asyncio.create_task(self._process_model_events()),
            asyncio.create_task(self._supervise_session()),
        ]
        
        logger.debug("Event loop started with %d background tasks", len(self.background_tasks))

    async def stop(self) -> None:
        """Gracefully shutdown and cleanup all resources."""
        if not self.active:
            return
            
        logger.debug("Stopping bidirectional event loop")
        self.active = False
        
        # Cancel all tasks
        for task in self.pending_tool_tasks.values():
            if not task.done():
                task.cancel()
        
        for task in self.background_tasks:
            if not task.done():
                task.cancel()
        
        # Wait for cancellations
        all_tasks = list(self.pending_tool_tasks.values()) + self.background_tasks
        if all_tasks:
            await asyncio.gather(*all_tasks, return_exceptions=True)
        
        # Close model session
        try:
            await self.model_session.close()
        except Exception as e:
            logger.warning("Error closing model session: %s", e)
            
        logger.debug("Event loop stopped - tools executed: %d", self.tool_count)

    def schedule_tool_execution(self, tool_use: ToolUse) -> None:
        """Create asyncio task for immediate tool execution (0ms scheduling)."""
        tool_name = tool_use.get("name")
        tool_id = tool_use.get("toolUseId")
        
        # Thread-safe counter increment
        current_tool_number = self.tool_count + 1
        self.tool_count = current_tool_number
        print(f"\nTool #{current_tool_number}: {tool_name}")
        
        logger.debug("Scheduling tool execution: %s (id: %s)", tool_name, tool_id)
        
        # Create task with UUID tracking
        task_id = str(uuid.uuid4())
        task = asyncio.create_task(self._execute_tool_with_strands(tool_use))
        self.pending_tool_tasks[task_id] = task
        
        def cleanup_task(completed_task: asyncio.Task) -> None:
            self.pending_tool_tasks.pop(task_id, None)
            if completed_task.cancelled():
                logger.debug("Tool task cancelled: %s", task_id)
            elif completed_task.exception():
                logger.error("Tool task error: %s - %s", task_id, completed_task.exception())
            else:
                logger.debug("Tool task completed: %s", task_id)
                
        task.add_done_callback(cleanup_task)

    async def handle_interruption(self) -> None:
        """Execute atomic interruption handling with race condition prevention.
        
        Always clears audio buffers for responsive interruption.
        Protects tool execution by not cancelling tools when they are running.
        """
        async with self.interruption_lock:
            if self.interrupted:
                logger.debug("Interruption already in progress")
                return

            logger.debug("Interruption detected")
            self.interrupted = True

            # Check if tools are currently executing
            active_tool_tasks = [task for task in self.pending_tool_tasks.values() if not task.done()]
            
            if active_tool_tasks:
                logger.debug("Tools are protected - %d tools currently executing", len(active_tool_tasks))
                # Don't cancel tools, but still clear audio for responsive interruption
            else:
                logger.debug("No active tools - full interruption handling")

            # Always clear audio queues for responsive interruption (regardless of tool status)
            cleared_count = 0
            while True:
                try:
                    self.audio_output_queue.get_nowait()
                    cleared_count += 1
                except asyncio.QueueEmpty:
                    break

            # Filter audio events from agent queue, preserve others
            temp_events = []
            try:
                while True:
                    event = self.agent._output_queue.get_nowait()
                    if event.get("audioOutput"):
                        cleared_count += 1
                    else:
                        temp_events.append(event)
            except asyncio.QueueEmpty:
                pass

            # Restore non-audio events
            for event in temp_events:
                self.agent._output_queue.put_nowait(event)

            self.interrupted = False
            
            if active_tool_tasks:
                logger.debug("Interruption handled (tools protected) - audio cleared: %d", cleared_count)
            else:
                logger.debug("Interruption handled (full) - audio cleared: %d", cleared_count)

    async def _process_model_events(self) -> None:
        """Process incoming provider event stream and dispatch to appropriate handlers."""
        logger.debug("Model events processor started")
        
        try:
            async for provider_event in self.model_session.receive_events():
                if not self.active:
                    break
                    
                if not isinstance(provider_event, dict):
                    continue
                    
                strands_event = provider_event
                
                # Handle interruptions
                if strands_event.get("interruptionDetected"):
                    logger.debug("Interruption detected from model")
                    await self.handle_interruption()
                    await self.agent._output_queue.put(strands_event)
                    continue
                
                # Schedule tool execution immediately
                if strands_event.get("toolUse"):
                    tool_name = strands_event["toolUse"].get("name")
                    logger.debug("Tool request received: %s", tool_name)
                    self.schedule_tool_execution(strands_event["toolUse"])
                    continue
                
                # Route audio to both queues
                if strands_event.get("audioOutput"):
                    await self.audio_output_queue.put(strands_event)
                    await self.agent._output_queue.put(strands_event)
                    continue
                
                # Forward text output
                if strands_event.get("textOutput"):
                    await self.agent._output_queue.put(strands_event)
                
                # Update conversation history (thread-safe)
                if strands_event.get("messageStop"):
                    logger.debug("Adding message to conversation history")
                    async with self.conversation_lock:
                        self.agent.messages.append(strands_event["messageStop"]["message"])
                
                # Handle user transcripts
                if (strands_event.get("textOutput") and 
                    strands_event["textOutput"].get("role") == "user"):
                    user_transcript = strands_event["textOutput"]["text"]
                    if user_transcript.strip():
                        user_message = {"role": "user", "content": user_transcript}
                        async with self.conversation_lock:
                            self.agent.messages.append(user_message)
                        logger.debug("User transcript added to history")
                        
        except Exception as e:
            logger.error("Model events processor error: %s", e)
            traceback.print_exc()
        finally:
            logger.debug("Model events processor stopped")

    async def _supervise_session(self) -> None:
        """Monitor background task health using event-driven completion waiting."""
        logger.debug("Session supervisor started")
        
        try:
            # Supervise tasks excluding self to avoid circular waiting
            tasks_to_supervise = [task for task in self.background_tasks if task != asyncio.current_task()]
            
            while self.active and tasks_to_supervise:
                # Wait for any task completion (deterministic vs polling)
                done, pending = await asyncio.wait(
                    tasks_to_supervise,
                    return_when=asyncio.FIRST_COMPLETED,
                    timeout=1.0  # Periodic active flag check
                )
                
                # Check for task failures
                for task in done:
                    if not task.cancelled():
                        exception = task.exception()
                        if exception:
                            logger.error("Background task failed: %s", exception)
                            self.active = False
                            break
                
                # Remove completed tasks from supervision list
                tasks_to_supervise = [task for task in tasks_to_supervise if not task.done()]
            
        except Exception as e:
            logger.error("Session supervisor error: %s", e)
        finally:
            logger.debug("Session supervisor stopped")

    async def _execute_tool_with_strands(self, tool_use: ToolUse) -> None:
        """Execute tool using Strands validation and execution pipeline."""
        tool_name = tool_use.get("name")
        tool_id = tool_use.get("toolUseId")
        
        logger.debug("Executing tool: %s (id: %s)", tool_name, tool_id)
        
        try:
            # Prepare for Strands validation system
            tool_message: Message = {"role": "assistant", "content": [{"toolUse": tool_use}]}
            tool_uses: list[ToolUse] = []
            tool_results: list[ToolResult] = []
            invalid_tool_use_ids: list[str] = []
            
            # Validate tools
            validate_and_prepare_tools(tool_message, tool_uses, tool_results, invalid_tool_use_ids)
            valid_tool_uses = [tu for tu in tool_uses if tu.get("toolUseId") not in invalid_tool_use_ids]
            
            if not valid_tool_uses:
                logger.warning("No valid tools after validation: %s", tool_name)
                return
            
            # Execute with agent context
            invocation_state = {
                "agent": self.agent,
                "model": self.agent.model,
                "messages": self.agent.messages,
                "system_prompt": self.agent.system_prompt,
            }
            
            cycle_trace = Trace("Bidirectional Tool Execution")
            tool_events = self.agent.tool_executor._execute(
                self.agent, valid_tool_uses, tool_results, cycle_trace, None, invocation_state
            )
            
            # Process tool event stream
            async for tool_event in tool_events:
                if isinstance(tool_event, ToolResultEvent):
                    tool_result = tool_event.tool_result
                    tool_use_id = tool_result.get("toolUseId")
                    await self.model_session.send_tool_result(tool_use_id, tool_result)
                    logger.debug("Tool result sent: %s", tool_use_id)
                elif isinstance(tool_event, ToolStreamEvent):
                    logger.debug("Tool stream event: %s", tool_event)
            
            # Update conversation history (thread-safe)
            if tool_results:
                tool_result_message: Message = {
                    "role": "user",
                    "content": [{"toolResult": result} for result in tool_results],
                }
                
                async with self.conversation_lock:
                    self.agent.messages.append(tool_result_message)
                    self.agent.hooks.invoke_callbacks(
                        MessageAddedEvent(agent=self.agent, message=tool_result_message)
                    )
                logger.debug("Tool result message added to history: %s", tool_name)
            
            logger.debug("Tool execution completed: %s", tool_name)
            
        except asyncio.CancelledError:
            logger.debug("Tool execution cancelled: %s (id: %s)", tool_name, tool_id)
            raise
        except Exception as e:
            logger.error("Tool execution error: %s - %s", tool_name, e)
            
            # Send error result to provider
            error_result: ToolResult = {
                "toolUseId": tool_id,
                "status": "error",
                "content": [{"text": f"Error: {str(e)}"}]
            }
            
            try:
                await self.model_session.send_tool_result(tool_id, error_result)
                logger.debug("Error result sent: %s", tool_id)
            except Exception:
                logger.error("Failed to send error result: %s", tool_id)


# Session lifecycle coordinator functions
async def start_bidirectional_connection(agent: "BidirectionalAgent") -> "BidirectionalEventLoop":
    """Initialize and start bidirectional streaming session."""
    logger.debug("Creating bidirectional connection")
    
    model_session = await agent.model.create_bidirectional_connection(
        system_prompt=agent.system_prompt,
        tools=agent.tool_registry.get_all_tool_specs(),
        messages=agent.messages
    )
    
    event_loop = BidirectionalEventLoop(model_session=model_session, agent=agent)
    await event_loop.start()
    
    logger.debug("Bidirectional connection created and started")
    return event_loop


async def stop_bidirectional_connection(event_loop: "BidirectionalEventLoop") -> None:
    """Terminate bidirectional streaming session and cleanup resources."""
    await event_loop.stop()