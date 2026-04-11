"""Built-in slash command handlers."""

from __future__ import annotations

import asyncio
import os
import sys
import time

from nanobot import __version__
from nanobot.bus.events import OutboundMessage
from nanobot.command.router import CommandContext, CommandRouter
from nanobot.utils.helpers import build_status_content
from nanobot.utils.restart import set_restart_notice_to_env


async def cmd_stop(ctx: CommandContext) -> OutboundMessage:
    """Cancel all active tasks and subagents for the session."""
    loop = ctx.loop
    msg = ctx.msg
    tasks = loop._active_tasks.pop(msg.session_key, [])
    cancelled = sum(1 for t in tasks if not t.done() and t.cancel())
    for t in tasks:
        try:
            await t
        except (asyncio.CancelledError, Exception):
            pass
    sub_cancelled = await loop.subagents.cancel_by_session(msg.session_key)
    total = cancelled + sub_cancelled
    content = f"Stopped {total} task(s)." if total else "No active task to stop."
    return OutboundMessage(
        channel=msg.channel, chat_id=msg.chat_id, content=content, metadata=dict(msg.metadata or {})
    )


async def cmd_restart(ctx: CommandContext) -> OutboundMessage:
    """Restart the process in-place via os.execv."""
    msg = ctx.msg
    set_restart_notice_to_env(channel=msg.channel, chat_id=msg.chat_id)

    async def _do_restart():
        await asyncio.sleep(1)
        os.execv(sys.executable, [sys.executable, "-m", "nanobot"] + sys.argv[1:])

    asyncio.create_task(_do_restart())
    return OutboundMessage(
        channel=msg.channel,
        chat_id=msg.chat_id,
        content="Restarting...",
        metadata=dict(msg.metadata or {}),
    )


async def cmd_status(ctx: CommandContext) -> OutboundMessage:
    """Build an outbound status message for a session."""
    loop = ctx.loop
    ctx_est = 0
    try:
        session = ctx.session or loop.sessions.get_session(ctx.key)
        ctx_est = loop.history_compressor.estimate_session_prompt_tokens(session)
    except Exception:
        pass
    return OutboundMessage(
        channel=ctx.msg.channel,
        chat_id=ctx.msg.chat_id,
        content=build_status_content(
            version=__version__,
            model=str(loop.models[0]) if loop.models else "unknown",
            start_time=getattr(loop, "_start_time", time.time()),
            last_usage=getattr(loop, "_last_usage", {}),
            context_window_tokens=loop.context_window_tokens,
            session_msg_count=len(loop.sessions.get_all_messages(ctx.key)),
            context_tokens_estimate=ctx_est,
        ),
        metadata={**dict(ctx.msg.metadata or {}), "render_as": "text"},
    )


async def cmd_compact(ctx: CommandContext) -> OutboundMessage:
    """Compress conversation history into a summary, then continue."""
    loop = ctx.loop
    loop.sessions.ensure_session(ctx.key)
    snapshot = loop.sessions.get_unconsolidated_messages(ctx.key)
    if snapshot:
        await loop.history_compressor.summarize(ctx.key, snapshot)
    return OutboundMessage(
        channel=ctx.msg.channel,
        chat_id=ctx.msg.chat_id,
        content="Conversation compressed.",
        metadata=dict(ctx.msg.metadata or {}),
    )


async def cmd_help(ctx: CommandContext) -> OutboundMessage:
    """Return available slash commands."""
    return OutboundMessage(
        channel=ctx.msg.channel,
        chat_id=ctx.msg.chat_id,
        content=build_help_text(),
        metadata={**dict(ctx.msg.metadata or {}), "render_as": "text"},
    )


def build_help_text() -> str:
    """Build canonical help text shared across channels."""
    lines = [
        "🐈 nanobot commands:",
        "/compact — Summarize conversation history",
        "/stop — Stop the current task",
        "/restart — Restart the bot",
        "/status — Show bot status",
        "/subagents [list|log <id>|kill <id>] — Manage background agents",
        "/help — Show available commands",
    ]
    return "\n".join(lines)


async def cmd_subagents(ctx: CommandContext) -> OutboundMessage:
    """Handle /subagents list|log|kill."""
    args = getattr(ctx, "args", None)
    if args is not None:
        args = args.strip()
    else:
        args = ""

    sub = ctx.loop.subagents

    if not args:
        return OutboundMessage(
            channel=ctx.msg.channel,
            chat_id=ctx.msg.chat_id,
            content="/subagents — Manage background agents\n\n"
            "  /subagents list      — List all subagents\n"
            "  /subagents log <id>  — Get logs for a subagent\n"
            "  /subagents kill <id> — Kill a running subagent",
        )

    if args == "list":
        return await _subagents_list(ctx, sub)

    if args.startswith("log "):
        id = args[4:].strip()
        return await _subagents_log(ctx, sub, id)

    if args.startswith("kill "):
        id = args[5:].strip()
        return await _subagents_kill(ctx, sub, id)

    return OutboundMessage(
        channel=ctx.msg.channel,
        chat_id=ctx.msg.chat_id,
        content="/subagents — Manage background agents\n\n"
        "  /subagents list      — List all subagents\n"
        "  /subagents log <id>  — Get logs for a subagent\n"
        "  /subagents kill <id> — Kill a running subagent",
    )


async def _subagents_list(ctx: CommandContext, sub) -> OutboundMessage:
    """List all subagents for the session."""
    session_key = ctx.key
    metas = sub.list_subagents(session_key)
    if not metas:
        content = "No subagents."
    else:
        lines = ["Subagents:"]
        for m in metas:
            status_icon = {
                "running": "🔄",
                "completed": "✅",
                "failed": "❌",
                "cancelled": "🚫",
                "interrupted": "⚡",
            }.get(m.status, "?")
            lines.append(f"  {status_icon} [{m.id[:8]}] {m.label} — {m.status}")
        content = "\n".join(lines)
    return OutboundMessage(channel=ctx.msg.channel, chat_id=ctx.msg.chat_id, content=content)


async def _subagents_log(ctx: CommandContext, sub, id: str) -> OutboundMessage:
    """Get log for a specific subagent by UUID."""
    meta = sub.get_by_id(id)
    if not meta:
        return OutboundMessage(
            channel=ctx.msg.channel,
            chat_id=ctx.msg.chat_id,
            content=f"No subagent with id '{id}'.",
        )

    sessions = ctx.loop.sessions
    try:
        messages = sessions.get_all_messages(meta.subagent_session_key)
    except (KeyError, Exception):
        messages = []

    if messages:
        from nanobot.agents.helpers import format_messages_for_text

        content = format_messages_for_text(messages)
    elif meta.result:
        content = f"[{meta.label}] Final result:\n{meta.result}"
    else:
        content = f"No logs available for '{meta.label}'."

    return OutboundMessage(channel=ctx.msg.channel, chat_id=ctx.msg.chat_id, content=content)


async def _subagents_kill(ctx: CommandContext, sub, id: str) -> OutboundMessage:
    """Kill a running subagent by UUID."""
    meta = sub.get_by_id(id)
    if not meta:
        return OutboundMessage(
            channel=ctx.msg.channel,
            chat_id=ctx.msg.chat_id,
            content=f"No subagent with id '{id}'.",
        )

    if meta.status != "running":
        return OutboundMessage(
            channel=ctx.msg.channel,
            chat_id=ctx.msg.chat_id,
            content=f"'{meta.label}' is already {meta.status}.",
        )

    killed = await sub.kill_by_id(id)
    verb = "killed" if killed else "could not kill"
    return OutboundMessage(
        channel=ctx.msg.channel,
        chat_id=ctx.msg.chat_id,
        content=f"Subagent '{meta.label}' {verb}.",
    )


def register_builtin_commands(router: CommandRouter) -> None:
    """Register the default set of slash commands."""
    router.priority("/stop", cmd_stop)
    router.priority("/restart", cmd_restart)
    router.priority("/status", cmd_status)
    router.exact("/compact", cmd_compact)
    router.exact("/help", cmd_help)
    router.exact("/subagents", cmd_subagents)
    router.prefix("/subagents ", cmd_subagents)
