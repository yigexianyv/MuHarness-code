
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


__all__ = ["Conversation", "ConversationMessageRecord"]
