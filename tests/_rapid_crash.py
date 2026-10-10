"""Worker functions that misbehave on purpose, for the crash-isolation tests (importable by a spawned worker)."""
import os
import signal
import time


def crash(arr):
    os.kill(os.getpid(), signal.SIGSEGV)


def sleep(arr):
    time.sleep(5)
    return []


def echo(arr):
    return [[[[0.0, 0.0], [10.0, 0.0], [10.0, 5.0], [0.0, 5.0]], f"shape {arr.shape[0]}x{arr.shape[1]}", 0.99]]
