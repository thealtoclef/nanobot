"""Event types for the message bus."""

from datetime import datetime
from typing import Any

from pydantic import BaseModel, Field


class InboundMessage(BaseModel):
    """Message received from a chat channel."""

    channel: str  # telegram, discord, slack, whatsapp
    sender_id: str  # User identifier
    chat_id: str  # Chat/channel identifier
    content: str  # Message text
    session_key: str  # Explicit session key, computed by channel
    timestamp: datetime = Field(default_factory=datetime.now)
    media: list[str] = Field(default_factory=list)  # Media URLs
    metadata: dict[str, Any] = Field(default_factory=dict)  # Channel-specific data


class OutboundMessage(BaseModel):
    """Message to send to a chat channel."""

    channel: str
    chat_id: str
    content: str
    reply_to: str | None = None
    media: list[str] = Field(default_factory=list)
    metadata: dict[str, Any] = Field(default_factory=dict)
