"""Heartbeat service - periodic agent wake-up to check for tasks."""

from __future__ import annotations

import asyncio
import json
import re
from pathlib import Path
from typing import Any, Callable, Coroutine

from loguru import logger


class HeartbeatService:
    """
    Periodic heartbeat service that wakes the agent to check for tasks.

    Phase 1 (decision): reads HEARTBEAT.md and asks the agent — via a
    structured text response — whether there are active tasks.

    Phase 2 (execution): only triggered when Phase 1 returns ``run``.  The
    ``on_execute`` callback runs the task through the full agent loop and
    returns the result to deliver.
    """

    def __init__(
        self,
        workspace: Path,
        agent: Any,
        on_execute: Callable[[str], Coroutine[Any, Any, str]] | None = None,
        on_notify: Callable[[str], Coroutine[Any, Any, None]] | None = None,
        interval_s: int = 30 * 60,
        enabled: bool = True,
        timezone: str | None = None,
    ):
        self.workspace = workspace
        self.agent = agent
        self.on_execute = on_execute
        self.on_notify = on_notify
        self.interval_s = interval_s
        self.enabled = enabled
        self.timezone = timezone
        self._running = False
        self._task: asyncio.Task | None = None

    @property
    def heartbeat_file(self) -> Path:
        return self.workspace / "HEARTBEAT.md"

    def _read_heartbeat_file(self) -> str | None:
        if self.heartbeat_file.exists():
            try:
                return self.heartbeat_file.read_text(encoding="utf-8")
            except Exception:
                return None
        return None

    async def _decide(self, content: str) -> tuple[str, str]:
        """Phase 1: ask agent to decide skip/run.

        Returns (action, tasks) where action is 'skip' or 'run'.
        The agent is asked to respond in a structured JSON format.
        """
        from nanobot.utils.helpers import current_time_str

        prompt = f"""Current Time: {current_time_str(self.timezone)}

Review the following HEARTBEAT.md and decide whether there are active tasks.

{content}

Respond ONLY with a JSON object with keys "action" (one of "skip" or "run") and optionally "tasks" (a summary string for run):
{{"action": "skip"}}  or  {{"action": "run", "tasks": "summary of active tasks"}}
"""

        response = await self.agent.run(
            user_message=prompt,
            session=None,
        )

        try:
            # Try to parse as JSON first
            parsed = json.loads(response.strip())
            action = parsed.get("action", "skip")
            tasks = parsed.get("tasks", "")
            return action, tasks
        except (json.JSONDecodeError, ValueError):
            pass

        # Fallback: try to extract from text
        action_match = re.search(r'"action"\s*:\s*"(\w+)"', response)
        tasks_match = re.search(r'"tasks"\s*:\s*"([^"]+)"', response)

        if action_match:
            action = action_match.group(1)
            tasks = tasks_match.group(1) if tasks_match else ""
            return action, tasks

        # Default to skip if we can't parse
        logger.warning("Heartbeat: could not parse decision response: {}", response[:100])
        return "skip", ""

    async def start(self) -> None:
        """Start the heartbeat service."""
        if not self.enabled:
            logger.info("Heartbeat disabled")
            return
        if self._running:
            logger.warning("Heartbeat already running")
            return

        self._running = True
        self._task = asyncio.create_task(self._run_loop())
        logger.info("Heartbeat started (every {}s)", self.interval_s)

    def stop(self) -> None:
        """Stop the heartbeat service."""
        self._running = False
        if self._task:
            self._task.cancel()
            self._task = None

    async def _run_loop(self) -> None:
        """Main heartbeat loop."""
        while self._running:
            try:
                await asyncio.sleep(self.interval_s)
                if self._running:
                    await self._tick()
            except asyncio.CancelledError:
                break
            except Exception as e:
                logger.error("Heartbeat error: {}", e)

    async def _evaluate(self, response: str, task_context: str) -> bool:
        """Phase 3: ask agent whether to notify the user about the result.

        Uses a lightweight agent call to decide if the response contains
        actionable information worth delivering to the user.
        """
        prompt = f"""You are a notification gate for a background agent.

## Original task
{task_context}

## Agent response
{response}

Respond ONLY with a JSON object with key "should_notify" (boolean) and optionally "reason" (string):
{{"should_notify": true, "reason": "brief explanation"}}
or to suppress:
{{"should_notify": false, "reason": "brief explanation"}}

Notify when the response contains actionable information, errors, completed deliverables,
or anything the user explicitly asked to be reminded about.

Suppress when the response is a routine status check, confirmation that everything is normal,
or essentially empty."""

        try:
            result = await self.agent.run(user_message=prompt, session=None)
            parsed = json.loads(result.strip())
            should_notify = bool(parsed.get("should_notify", True))
            reason = parsed.get("reason", "")
            logger.info("Heartbeat evaluation: should_notify={}, reason={}", should_notify, reason)
            return should_notify
        except (json.JSONDecodeError, ValueError):
            pass

        # Fallback: try to extract from text
        match = re.search(r'"should_notify"\s*:\s*(true|false)', result, re.IGNORECASE)
        if match:
            return match.group(1).lower() == "true"

        # Default to notify on parse failure
        logger.warning("Heartbeat: could not parse evaluation response: {}", result[:100])
        return True

    async def _tick(self) -> None:
        """Execute a single heartbeat tick."""
        content = self._read_heartbeat_file()
        if not content:
            logger.debug("Heartbeat: HEARTBEAT.md missing or empty")
            return

        logger.info("Heartbeat: checking for tasks...")

        try:
            action, tasks = await self._decide(content)

            if action != "run":
                logger.info("Heartbeat: OK (nothing to report)")
                return

            logger.info("Heartbeat: tasks found, executing...")
            if self.on_execute:
                response = await self.on_execute(tasks)

                if response:
                    should_notify = await self._evaluate(response, tasks)
                    if should_notify and self.on_notify:
                        logger.info("Heartbeat: completed, delivering response")
                        await self.on_notify(response)
                    else:
                        logger.info("Heartbeat: silenced by evaluation")
        except Exception:
            logger.exception("Heartbeat execution failed")

    async def trigger_now(self) -> str | None:
        """Manually trigger a heartbeat."""
        content = self._read_heartbeat_file()
        if not content:
            return None
        action, tasks = await self._decide(content)
        if action != "run" or not self.on_execute:
            return None
        return await self.on_execute(tasks)
