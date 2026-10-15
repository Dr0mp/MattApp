from __future__ import annotations

import subprocess
import tempfile
import os
from threading import Event


class JobCancelled(Exception):
    """A requested cancellation, rather than an inference failure."""


class Cancellation:
    def __init__(self) -> None:
        self._event = Event()

    @property
    def requested(self) -> bool:
        return self._event.is_set()

    def cancel(self) -> None:
        self._event.set()

    def wait(self, timeout: float) -> bool:
        """Wait briefly for retries; return immediately when Stop is requested."""
        return self._event.wait(timeout)

    def check(self) -> None:
        if self.requested:
            raise JobCancelled()

    def run_process(self, command: list[str], capture_output: bool = False, on_poll=None) -> bytes | None:
        """Reap the child before callers clean up its temporary input/output files."""
        self.check()
        with tempfile.TemporaryFile() as output, tempfile.TemporaryFile() as errors, subprocess.Popen(
            command, stdout=output if capture_output else None, stderr=errors,
            creationflags=(subprocess.CREATE_NO_WINDOW | 0x4) if os.name == "nt" else 0,
        ) as process:
            from .model_session import OwnedJob
            try:owned=OwnedJob(process)
            except Exception:
                process.terminate();process.wait();raise
            try:
                while True:
                    if on_poll:
                        on_poll()
                    self.check()
                    try:
                        returncode = process.wait(timeout=0.1)
                        break
                    except subprocess.TimeoutExpired:
                        pass
                self.check()
                if returncode:
                    errors.seek(0)
                    raise subprocess.CalledProcessError(returncode, command, stderr=errors.read().decode("utf-8", errors="replace"))
                if capture_output:
                    output.seek(0)
                    return output.read()
            finally:
                if process.poll() is None:
                    process.terminate()
                    try:
                        process.wait(timeout=3)
                    except subprocess.TimeoutExpired:
                        process.kill()
                        process.wait()
                owned.close()
