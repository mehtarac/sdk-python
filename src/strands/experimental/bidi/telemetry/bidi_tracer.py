"""Bidirectional agent telemetry tracer."""

from typing import Optional

from opentelemetry import trace as trace_api

from ....telemetry.tracer import Tracer, serialize
from ....types.content import Message
from ..types.events import BidiTranscriptStreamEvent


class BidiTracer(Tracer):
    """Tracer for bidirectional agent operations."""

    def __init__(self):
        """Initialize BidiTracer."""
        super().__init__()
        self._model_span: Optional[trace_api.Span] = None

    def end_model_span(self, message: Message, all_messages, usage=None, metrics=None):
        """End model invocation span."""
        if self._model_span:
            self._model_span.set_attribute(
                "gen_ai.input.messages", serialize([msg for msg in all_messages if msg["role"] == "user"][-1:])
            )
            self._model_span.set_attribute("gen_ai.output.messages", serialize([message]))

            self.end_model_invoke_span(
                self._model_span,
                message=message,
                usage=usage or {"inputTokens": 0, "outputTokens": 0, "totalTokens": 0},
                metrics=metrics or {},
                stop_reason="end_turn",
            )
            self._model_span = None

    def update_session_span(self, session_span, event: BidiTranscriptStreamEvent, message: Message, all_messages):
        """Update session span with conversation messages."""
        if not session_span:
            return

        if event["role"] == "user":
            session_span.set_attribute("gen_ai.input.messages", serialize(all_messages))
        elif event["role"] == "assistant":
            session_span.set_attribute("gen_ai.output.messages", serialize([message]))

    def update_session_turn_counts(self, session_span, user_turns: int, assistant_turns: int):
        """Update session span with turn counts."""
        if session_span:
            session_span.set_attribute("strands.session.user_turns", str(user_turns))
            session_span.set_attribute("strands.session.assistant_turns", str(assistant_turns))

    def handle_transcript_event(self, event, agent, loop_span):
        """Handle transcript stream events with tracing."""
        if event["role"] == "user" and event["is_final"]:
            if not self._model_span:
                self._model_span = self.start_model_invoke_span(
                    messages=agent.messages,
                    parent_span=loop_span,
                    model_id=getattr(agent.model, "model_id", None),
                )

        if event["is_final"]:
            message = {"role": event["role"], "content": [{"text": event["text"]}]}
            agent.messages.append(message)

            self.update_session_span(agent._session_span, event, message, agent.messages)

            # Update turn counts
            if event["role"] == "user":
                agent._conversation_state["user_turns"] += 1
            elif event["role"] == "assistant":
                agent._conversation_state["assistant_turns"] += 1

            self.update_session_turn_counts(
                agent._session_span,
                agent._conversation_state["user_turns"],
                agent._conversation_state["assistant_turns"],
            )

            if event["role"] == "assistant":
                self.end_model_span(message, agent.messages)

    def handle_tool_use_event(self, event, agent, loop_span):
        """Handle tool use stream events with tracing."""
        tool_use = event["current_tool_use"]
        agent._conversation_state["tool_executions"] += 1

        if not self._model_span:
            self._model_span = self.start_model_invoke_span(
                messages=agent.messages,
                parent_span=loop_span,
                model_id=getattr(agent.model, "model_id", None),
            )

        message = {"role": "assistant", "content": [{"toolUse": tool_use}]}
        agent.messages.append(message)
        return tool_use

    @property
    def current_model_span(self):
        """Get current model span."""
        return self._model_span


_bidi_tracer_instance = None


def get_bidi_tracer() -> BidiTracer:
    """Get or create the global bidi tracer instance."""
    global _bidi_tracer_instance
    if not _bidi_tracer_instance:
        _bidi_tracer_instance = BidiTracer()
    return _bidi_tracer_instance
