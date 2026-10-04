"""S03 的独立进程入口。只执行既定会话请求，不执行模型产物。"""

from __future__ import annotations

import asyncio
import hashlib
import json
import os
import sys
from pathlib import Path

from .spec import Case, Turn, Variant
from .stage import Stage, StagePaths, install_approval_policy, patched_env
from .v1 import _secrets, live_factory, write_json
from .v2_driver import dispatch_turn


async def run(payload: dict) -> None:
    case, variant = (
        Case.model_validate(payload["case"]),
        Variant.model_validate(payload["variant"]),
    )
    paths = StagePaths.under(Path(payload["root"]).resolve())
    output = Path(payload["result_path"]).resolve()
    if not output.is_relative_to(paths.data.resolve()):
        raise ValueError("子进程结果路径越界")
    if payload["scripted"]:
        from .selftest.scripted import answer, scripted_factory

        def reply(request):
            user = [
                message.content or ""
                for message in request.messages
                if message.role.value == "user"
            ]
            if len(user) > 1:
                present = any(
                    "PULSAR" in text and "73000" in text for text in user[:-1]
                )
                return answer(
                    json.dumps(
                        {
                            "project": "PULSAR" if present else "MISSING",
                            "budget": 73000 if present else 0,
                        }
                    )
                )
            return answer("收到")

        factory = scripted_factory(reply)
    else:
        factory = live_factory(
            provider=payload["provider"],
            model=payload["model"],
            non_stream=payload["non_stream"],
            api_retries=payload["api_retries"],
        )
    with patched_env({**variant.env, **case.env}):
        app = factory(paths, variant)
        stage = Stage(case=case, variant=variant, paths=paths, app=app, factory=factory)
        try:
            await app.start()
            install_approval_policy(app, case.approvals)
            history = await app.conversation_store.load_messages(
                payload["conversation_id"]
            )
            serialized = [message.model_dump(mode="json") for message in history]
            turn = await dispatch_turn(
                stage, payload["conversation_id"], Turn.model_validate(payload["turn"])
            )
            adapter = app.registry.get(app.provider)
            data = {
                "worker": {
                    "pid": os.getpid(),
                    "conversation_id": payload["conversation_id"],
                    "database": str(app.database),
                    "history_before_count": len(history),
                    "history_before_sha256": hashlib.sha256(
                        json.dumps(serialized, sort_keys=True).encode()
                    ).hexdigest(),
                    "run_id": turn.run_id,
                    "scripted": payload["scripted"],
                },
                "turn": turn.model_dump(mode="json"),
                "api_requests": list(getattr(adapter, "events", [])),
            }
            write_json(output, data, secrets=_secrets(app))
        finally:
            await app.close()


def main() -> None:
    payload = json.loads(Path(sys.argv[1]).read_text(encoding="utf-8"))
    asyncio.run(run(payload))


if __name__ == "__main__":
    main()
