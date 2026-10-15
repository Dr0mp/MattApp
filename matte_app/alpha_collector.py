"""Read atomic model outputs without treating transient Windows locks as failures."""
from pathlib import Path
import time
from PIL import Image


class AlphaCollector:
    def __init__(self, folder, consume, token, timeout=5.0, retry_interval=0.05):
        self.folder = Path(folder)
        self.consume = consume
        self.token = token
        self.timeout = timeout
        self.retry_interval = retry_interval
        self.delivered = 0
        self._blocked_since = None
        self._consume_failed = False

    def __call__(self):
        if self._consume_failed:
            return
        while True:
            path = self.folder / f'alpha_{self.delivered:06d}.png'
            if not path.exists():
                self._blocked_since = None
                return
            try:
                with Image.open(path) as image:
                    alpha = image.convert('L')
            except PermissionError as error:
                # A complete, renamed PNG can briefly still be exclusively locked
                # on Windows. Return to the process loop so Stop stays responsive.
                if self.token.requested:
                    return
                now = time.monotonic()
                if self._blocked_since is None:
                    self._blocked_since = now
                if now - self._blocked_since >= self.timeout:
                    raise RuntimeError(f'Windows kept the matte file locked for {self.timeout:g} seconds: {path}. Any completed exports were retained; retry the job.') from error
                return
            self._blocked_since = None
            try:
                self.consume(self.delivered, alpha)
            except BaseException:
                self._consume_failed = True
                raise
            self.delivered += 1

    def finish(self):
        """Drain the final outputs after the writer has exited, with bounded retries."""
        while True:
            self()
            if self._blocked_since is None or self.token.requested:
                return
            # Cancellation wakes this wait immediately. The process has already
            # been reaped, so cancellation can safely leave unread temporary files.
            if self.token.wait(self.retry_interval):
                return
