"""Invoke the unmodified official CLI, preserving patch LF on Windows hosts."""

from __future__ import annotations

import os
import runpy
from pathlib import Path


def main():
    if os.name == "nt":
        original = Path.write_text

        def write_lf(self, data, encoding=None, errors=None, newline=None):
            # The official CLI writes model_patch using Path.write_text(). Its
            # Windows CRLF translation makes an LF repository reject that patch.
            return original(
                self,
                data,
                encoding=encoding or "utf-8",
                errors=errors,
                newline="\n" if newline is None else newline,
            )

        Path.write_text = write_lf
    runpy.run_module("swebench.harness.run_evaluation", run_name="__main__")


if __name__ == "__main__":
    main()
