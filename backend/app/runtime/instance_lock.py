from __future__ import annotations

import os
from pathlib import Path
from typing import BinaryIO

if os.name == "nt":
    import msvcrt
else:
    import fcntl


class DatabaseInstanceLock:
    """Hold one backend's ownership until shutdown; the OS releases it on exit."""

    def __init__(self, database: Path) -> None:
        self.path = database.with_name(database.name + ".instance.lock")
        self._file: BinaryIO | None = None

    def acquire(self) -> None:
        if self._file is not None:
            raise RuntimeError("数据库实例锁已由当前后端持有")
        self.path.parent.mkdir(parents=True, exist_ok=True)
        file = self.path.open("a+b")
        try:
            if os.fstat(file.fileno()).st_size == 0:
                file.write(b"\0")
                file.flush()
            file.seek(0)
            if os.name == "nt":
                msvcrt.locking(file.fileno(), msvcrt.LK_NBLCK, 1)
            else:
                fcntl.flock(file.fileno(), fcntl.LOCK_EX | fcntl.LOCK_NB)
        except OSError as exc:
            file.close()
            raise RuntimeError(
                f"数据库已被其他后端占用：{self.path}。请先停止旧后端再启动。"
            ) from exc
        except BaseException:
            file.close()
            raise
        self._file = file

    def release(self) -> None:
        # Keep the file: deleting it could let another process lock a different inode.
        if self._file is not None:
            self._file.close()
            self._file = None
