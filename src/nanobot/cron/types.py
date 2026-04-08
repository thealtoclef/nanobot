"""Cron types."""

from typing import Literal

from pydantic import BaseModel, Field


class CronSchedule(BaseModel):
    """Schedule definition for a cron job."""

    kind: Literal["at", "every", "cron"]
    # For "at": timestamp in ms
    at_ms: int | None = None
    # For "every": interval in ms
    every_ms: int | None = None
    # For "cron": cron expression (e.g. "0 9 * * *")
    expr: str | None = None
    # Timezone for cron expressions
    tz: str | None = None


class CronPayload(BaseModel):
    """What to do when the job runs."""

    kind: Literal["system_event", "agent_turn"] = "agent_turn"
    message: str = ""
    # Deliver response to channel
    deliver: bool = False
    channel: str | None = None  # e.g. "whatsapp"
    to: str | None = None  # e.g. phone number


class CronRunRecord(BaseModel):
    """A single execution record for a cron job."""

    run_at_ms: int
    status: Literal["ok", "error", "skipped"]
    duration_ms: int = 0
    error: str | None = None


class CronJobState(BaseModel):
    """Runtime state of a job."""

    next_run_at_ms: int | None = None
    last_run_at_ms: int | None = None
    last_status: Literal["ok", "error", "skipped"] | None = None
    last_error: str | None = None
    run_history: list[CronRunRecord] = Field(default_factory=list)


class CronJob(BaseModel):
    """A scheduled job."""

    id: str
    name: str
    enabled: bool = True
    schedule: CronSchedule = Field(default_factory=lambda: CronSchedule(kind="every"))
    payload: CronPayload = Field(default_factory=CronPayload)
    state: CronJobState = Field(default_factory=CronJobState)
    created_at_ms: int = 0
    updated_at_ms: int = 0
    delete_after_run: bool = False


class CronStore(BaseModel):
    """Persistent store for cron jobs."""

    version: int = 1
    jobs: list[CronJob] = Field(default_factory=list)
