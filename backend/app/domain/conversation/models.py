
from __future__ import annotations

from datetime import datetime

from pydantic import BaseModel, ConfigDict, Field

from app.models.types import Message


class Conversation(BaseModel):

    model_config = ConfigDict(extra="forbid")

    id: str
    title: str
    created_at: datetime
    updated_at: datetime
    message_count: int = Field(default=0, ge=0)


class ConversationMessageRecord(BaseModel):

    model_config = ConfigDict(extra="forbid", frozen=True)

    sequence: int = Field(ge=0)
    message: Message
    created_at: datetime


class ConversationConstraints(BaseModel):
    """会话的"必须记住的事项"：每次请求都附带，不参与压缩。"""

    model_config = ConfigDict(extra="forbid", frozen=True)

    conversation_id: str
    text: str = ""
    revision: int = Field(default=0, ge=0)
    updated_at: datetime | None = None


__all__ = ["Conversation", "ConversationConstraints", "ConversationMessageRecord"]
