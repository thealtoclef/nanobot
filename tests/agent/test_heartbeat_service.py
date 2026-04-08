"""HeartbeatService tests using PydanticAI FunctionModel pattern.

FunctionModel intercepts at the model layer — it receives list[ModelMessage] and AgentInfo,
returns ModelResponse. This properly validates that NanobotAgent.run() passes correct
message structure to the model, exercising the real PydanticAI agent pipeline.
"""

import asyncio
from pathlib import Path

import pytest

from nanobot.agent import NanobotAgent
from nanobot.heartbeat.service import HeartbeatService

# PydanticAI FunctionModel imports
from pydantic_ai.models.function import AgentInfo, FunctionModel
from pydantic_ai import ModelResponse, TextPart, ModelMessage
from pydantic_ai.messages import ModelRequest, UserPromptPart


# ---------------------------------------------------------------------------
# Helper: build NanobotAgent with FunctionModel
# ---------------------------------------------------------------------------


def create_agent(response: str, tmp_path: Path) -> NanobotAgent:
    """Create a NanobotAgent that always returns the same JSON response."""

    def single_response(messages: list[ModelMessage], info: AgentInfo) -> ModelResponse:
        return ModelResponse(parts=[TextPart(content=response)])

    return NanobotAgent(workspace=tmp_path, models=[FunctionModel(single_response)])


def create_multi_response_agent(responses: list[str], tmp_path: Path) -> NanobotAgent:
    """Create a NanobotAgent that returns sequential responses from a list."""
    responses = list(responses)
    response_idx = [0]

    def multi_response(messages: list[ModelMessage], info: AgentInfo) -> ModelResponse:
        idx = response_idx[0]
        response_idx[0] += 1
        content = responses[idx] if idx < len(responses) else responses[-1]
        return ModelResponse(parts=[TextPart(content=content)])

    return NanobotAgent(workspace=tmp_path, models=[FunctionModel(multi_response)])


def _extract_prompt_text(messages: list[ModelMessage]) -> str:
    """Extract user prompt text from a list of ModelMessages."""
    for msg in messages:
        if isinstance(msg, ModelRequest):
            for part in msg.parts:
                if isinstance(part, UserPromptPart):
                    content = part.content
                    # UserPromptPart.content can be str | TextContent | list | etc.
                    # If it's a string, return it directly
                    if isinstance(content, str):
                        return content
                    # If it's a list, try to extract string content
                    if isinstance(content, list):
                        for item in content:
                            if isinstance(item, str):
                                return item
    return ""


# ---------------------------------------------------------------------------
# Tests
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_start_is_idempotent(tmp_path) -> None:
    # No agent calls needed for this test - just start/stop service
    agent = create_agent('{"action": "skip"}', tmp_path)
    service = HeartbeatService(
        workspace=tmp_path,
        agent=agent,
        interval_s=9999,
        enabled=True,
    )

    await service.start()
    first_task = service._task
    await service.start()

    assert service._task is first_task

    service.stop()
    await asyncio.sleep(0)


@pytest.mark.asyncio
async def test_decide_returns_skip_when_no_run_action(tmp_path) -> None:
    agent = create_agent('{"action": "skip"}', tmp_path)
    service = HeartbeatService(
        workspace=tmp_path,
        agent=agent,
    )

    action, tasks = await service._decide("heartbeat content")
    assert action == "skip"
    assert tasks == ""


@pytest.mark.asyncio
async def test_decide_returns_run_with_tasks(tmp_path) -> None:
    agent = create_agent('{"action": "run", "tasks": "check open tasks"}', tmp_path)
    service = HeartbeatService(
        workspace=tmp_path,
        agent=agent,
    )

    action, tasks = await service._decide("heartbeat content")
    assert action == "run"
    assert tasks == "check open tasks"


@pytest.mark.asyncio
async def test_trigger_now_executes_when_decision_is_run(tmp_path) -> None:
    (tmp_path / "HEARTBEAT.md").write_text("- [ ] do thing", encoding="utf-8")

    agent = create_agent('{"action": "run", "tasks": "check open tasks"}', tmp_path)

    called_with: list[str] = []

    async def _on_execute(tasks: str) -> str:
        called_with.append(tasks)
        return "done"

    service = HeartbeatService(
        workspace=tmp_path,
        agent=agent,
        on_execute=_on_execute,
    )

    result = await service.trigger_now()
    assert result == "done"
    assert called_with == ["check open tasks"]


@pytest.mark.asyncio
async def test_trigger_now_returns_none_when_decision_is_skip(tmp_path) -> None:
    (tmp_path / "HEARTBEAT.md").write_text("- [ ] do thing", encoding="utf-8")

    agent = create_agent('{"action": "skip"}', tmp_path)

    async def _on_execute(tasks: str) -> str:
        return tasks

    service = HeartbeatService(
        workspace=tmp_path,
        agent=agent,
        on_execute=_on_execute,
    )

    assert await service.trigger_now() is None


@pytest.mark.asyncio
async def test_tick_notifies_on_execution(tmp_path) -> None:
    """Phase 1 run -> Phase 2 execute -> Phase 3 evaluate -> notify."""
    (tmp_path / "HEARTBEAT.md").write_text("- [ ] check deployments", encoding="utf-8")

    # First response for _decide(), second for _evaluate()
    agent = create_multi_response_agent(
        [
            '{"action": "run", "tasks": "check deployments"}',
            '{"should_notify": true, "reason": "actionable error"}',
        ],
        tmp_path,
    )

    executed: list[str] = []
    notified: list[str] = []

    async def _on_execute(tasks: str) -> str:
        executed.append(tasks)
        return "deployment failed on staging"

    async def _on_notify(response: str) -> None:
        notified.append(response)

    service = HeartbeatService(
        workspace=tmp_path,
        agent=agent,
        on_execute=_on_execute,
        on_notify=_on_notify,
    )

    await service._tick()
    assert executed == ["check deployments"]
    assert notified == ["deployment failed on staging"]


@pytest.mark.asyncio
async def test_tick_silences_when_evaluation_says_no(tmp_path) -> None:
    """Phase 1 run -> Phase 2 execute -> Phase 3 evaluate -> silent (no notify)."""
    (tmp_path / "HEARTBEAT.md").write_text("- [ ] check status", encoding="utf-8")

    agent = create_multi_response_agent(
        [
            '{"action": "run", "tasks": "check status"}',
            '{"should_notify": false, "reason": "routine status check"}',
        ],
        tmp_path,
    )

    async def _on_execute(tasks: str) -> str:
        return "everything is fine"

    notified: list[str] = []

    async def _on_notify(response: str) -> None:
        notified.append(response)

    service = HeartbeatService(
        workspace=tmp_path,
        agent=agent,
        on_execute=_on_execute,
        on_notify=_on_notify,
    )

    await service._tick()
    assert notified == []


@pytest.mark.asyncio
async def test_decide_prompt_includes_current_time(tmp_path) -> None:
    """Phase 1 user prompt must contain current time so the LLM can judge task urgency."""
    captured_prompts: list[str] = []

    def capture_prompt(messages: list[ModelMessage], info: AgentInfo) -> ModelResponse:
        captured_prompts.append(_extract_prompt_text(messages))
        return ModelResponse(parts=[TextPart(content='{"action": "skip"}')])

    agent = NanobotAgent(workspace=tmp_path, models=[FunctionModel(capture_prompt)])

    service = HeartbeatService(
        workspace=tmp_path,
        agent=agent,
    )

    await service._decide("- [ ] check servers at 10:00 UTC")

    assert len(captured_prompts) == 1
    assert "Current Time:" in captured_prompts[0]


@pytest.mark.asyncio
async def test_decide_handles_non_json_response(tmp_path) -> None:
    """Non-JSON response should fall back to skip."""
    agent = create_agent("This is not JSON", tmp_path)
    service = HeartbeatService(
        workspace=tmp_path,
        agent=agent,
    )

    action, tasks = await service._decide("heartbeat content")
    assert action == "skip"
    assert tasks == ""


@pytest.mark.asyncio
async def test_decide_handles_missing_action_key(tmp_path) -> None:
    """JSON without action key should fall back to skip."""
    agent = create_agent('{"other": "value"}', tmp_path)
    service = HeartbeatService(
        workspace=tmp_path,
        agent=agent,
    )

    action, tasks = await service._decide("heartbeat content")
    assert action == "skip"
    assert tasks == ""


# ---------------------------------------------------------------------------
# Notification gate tests (_evaluate)
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_evaluate_returns_true_when_should_notify_is_true(tmp_path) -> None:
    """Phase 3: evaluation response with should_notify=true returns True."""
    agent = create_agent('{"should_notify": true, "reason": "actionable error"}', tmp_path)
    service = HeartbeatService(workspace=tmp_path, agent=agent)

    result = await service._evaluate("deployment failed on staging", "check deployments")

    assert result is True


@pytest.mark.asyncio
async def test_evaluate_returns_false_when_should_notify_is_false(tmp_path) -> None:
    """Phase 3: evaluation response with should_notify=false returns False."""
    agent = create_agent('{"should_notify": false, "reason": "routine check"}', tmp_path)
    service = HeartbeatService(workspace=tmp_path, agent=agent)

    result = await service._evaluate("all systems normal", "check status")

    assert result is False


@pytest.mark.asyncio
async def test_evaluate_falls_back_to_true_on_parse_error(tmp_path) -> None:
    """Non-JSON evaluation response falls back to should_notify=True."""
    agent = create_agent("I think you should know about this", tmp_path)
    service = HeartbeatService(workspace=tmp_path, agent=agent)

    result = await service._evaluate("some response", "some task")

    assert result is True


@pytest.mark.asyncio
async def test_evaluate_extracts_from_text_when_json_fails(tmp_path) -> None:
    """Text-only response with should_notify key is extracted."""
    agent = create_agent('{"should_notify": false, "reason": "all clear"}', tmp_path)
    service = HeartbeatService(workspace=tmp_path, agent=agent)

    result = await service._evaluate("everything is fine", "routine check")

    assert result is False


@pytest.mark.asyncio
async def test_evaluate_includes_task_context_in_prompt(tmp_path) -> None:
    """The task context is passed to the evaluation prompt."""
    captured_prompts: list[str] = []

    def capture_prompt(messages: list[ModelMessage], info: AgentInfo) -> ModelResponse:
        captured_prompts.append(_extract_prompt_text(messages))
        return ModelResponse(parts=[TextPart(content='{"should_notify": false, "reason": "ok"}')])

    agent = NanobotAgent(workspace=tmp_path, models=[FunctionModel(capture_prompt)])

    service = HeartbeatService(workspace=tmp_path, agent=agent)

    await service._evaluate("the result", "my specific task")

    assert len(captured_prompts) == 1
    assert "my specific task" in captured_prompts[0]
    assert "the result" in captured_prompts[0]
