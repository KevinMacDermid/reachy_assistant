#!/usr/bin/env python3
import os
import signal
import subprocess
import sys
from pathlib import Path
from time import sleep

from main import ZERO_AUDIO_EXIT_CODE
import time



def _terminate_process_group(proc: subprocess.Popen, name: str, timeout_s: float = 5.0) -> None:
    if proc.poll() is not None:
        return
    try:
        os.killpg(proc.pid, signal.SIGTERM)
    except ProcessLookupError:
        return
    try:
        proc.wait(timeout=timeout_s)
    except subprocess.TimeoutExpired:
        os.killpg(proc.pid, signal.SIGKILL)
        proc.wait()


def _monitor_main(main_proc: subprocess.Popen) -> int:
    assert main_proc.stdout is not None
    for line in iter(main_proc.stdout.readline, ""):
        sys.stdout.write(line)
        sys.stdout.flush()
    return main_proc.wait()


def _run_reset(root_dir: Path) -> int:
    return subprocess.run(
        ["/bin/bash", str(root_dir / "script" / "reset_internal_hub.sh")],
        cwd=str(root_dir),
    ).returncode


def main() -> int:
    root_dir = Path(__file__).resolve().parents[0]

    while True:
        # Want "--no-wake-up-on-start", but found it wouldn't wake up later
        daemon_proc = subprocess.Popen(
            [
                "uv", "run", "reachy-mini-daemon", "--headless"
            ],
            cwd=str(root_dir),
            start_new_session=True,
        )
        time.sleep(20)
        main_proc = subprocess.Popen(
            ["uv", "run", "python", "main.py"],
            cwd=str(root_dir),
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            text=True,
            bufsize=1,
            start_new_session=True,
        )

        try:
            main_exit_code = _monitor_main(main_proc)
        except KeyboardInterrupt:
            _terminate_process_group(main_proc, "main.py")
            _terminate_process_group(daemon_proc, "reachy-mini-daemon")
            return 130
        finally:
            if main_proc.stdout is not None:
                main_proc.stdout.close()

        if main_exit_code == ZERO_AUDIO_EXIT_CODE:
            _terminate_process_group(main_proc, "main.py")
            _terminate_process_group(daemon_proc, "reachy-mini-daemon")
            reset_code = _run_reset(root_dir)
            if reset_code != 0:
                print(f"reset_internal_hub.sh failed with exit code {reset_code}", file=sys.stderr)
                return reset_code
            print("Waiting before restarting daemon")
            sleep(20)
            continue

        _terminate_process_group(daemon_proc, "reachy-mini-daemon")
        return main_exit_code or 0


if __name__ == "__main__":
    raise SystemExit(main())
