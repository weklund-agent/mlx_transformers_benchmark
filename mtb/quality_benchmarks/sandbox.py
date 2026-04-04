"""Subprocess-based code execution sandbox for quality benchmarks.

Executes Python code in an isolated subprocess with configurable timeout,
captures stdout/stderr/exit code, strips markdown code fences, and cleans
up temp files regardless of outcome.

Thread-safe: each call creates its own temp file and subprocess, so
concurrent invocations don't interfere with each other.
"""

import os
import re
import subprocess
import tempfile
import time
from dataclasses import dataclass
from typing import Optional


@dataclass
class SandboxResult:
    """Result of a sandboxed code execution.

    Fields:
        success: True if code ran without error (exit code 0).
        stdout: Captured standard output from the subprocess.
        stderr: Captured standard error from the subprocess.
        error_type: Classification of the error if any (e.g. 'timeout',
            'syntax_error', 'runtime_error'), or None on success.
        return_code: Process exit code (0 = success, non-zero = error,
            -1 for timeout).
        execution_time_sec: Wall-clock execution time in seconds.
    """

    success: bool
    stdout: str
    stderr: str
    error_type: Optional[str]
    return_code: int
    execution_time_sec: float


def _strip_markdown_fences(code: str) -> str:
    """Strip markdown code fences from code, extracting fenced content.

    Handles:
    - ```python ... ``` blocks
    - ``` ... ``` blocks (no language tag)
    - Multiple fenced blocks (content concatenated)
    - Code with surrounding prose (only fenced content extracted)
    - Code without any fences (returned as-is)
    """
    # Pattern matches ```<optional-lang>\n...\n```
    fence_pattern = re.compile(r"```(?:\w*)\s*\n(.*?)```", re.DOTALL)
    matches = fence_pattern.findall(code)
    if matches:
        return "\n".join(matches)
    return code


def _classify_error(stderr: str) -> str:
    """Classify the error type from stderr output.

    Returns one of: 'syntax_error', 'runtime_error'.
    """
    if "SyntaxError" in stderr:
        return "syntax_error"
    return "runtime_error"


def execute_code(code: str, timeout: int = 10) -> SandboxResult:
    """Execute Python code in a subprocess sandbox.

    Args:
        code: Python source code to execute. Markdown code fences are
            stripped automatically. Empty/whitespace-only code is handled
            gracefully.
        timeout: Maximum execution time in seconds. Default is 10.
            Code exceeding this limit is killed.

    Returns:
        SandboxResult with execution outcome, captured output, and timing.
    """
    # Strip markdown fences
    code = _strip_markdown_fences(code)

    # Handle empty/whitespace-only code
    if not code.strip():
        return SandboxResult(
            success=True,
            stdout="",
            stderr="",
            error_type=None,
            return_code=0,
            execution_time_sec=0.0,
        )

    # Create temp file for the code
    tmp_fd, tmp_path = tempfile.mkstemp(prefix="sandbox_", suffix=".py")
    try:
        # Write code to temp file
        with os.fdopen(tmp_fd, "w") as f:
            f.write(code)

        # Execute in subprocess with timeout
        start_time = time.monotonic()
        try:
            result = subprocess.run(
                ["python", tmp_path],
                capture_output=True,
                text=True,
                timeout=timeout,
            )
            elapsed = time.monotonic() - start_time

            if result.returncode == 0:
                return SandboxResult(
                    success=True,
                    stdout=result.stdout,
                    stderr=result.stderr,
                    error_type=None,
                    return_code=0,
                    execution_time_sec=elapsed,
                )
            else:
                return SandboxResult(
                    success=False,
                    stdout=result.stdout,
                    stderr=result.stderr,
                    error_type=_classify_error(result.stderr),
                    return_code=result.returncode,
                    execution_time_sec=elapsed,
                )

        except subprocess.TimeoutExpired as e:
            elapsed = time.monotonic() - start_time
            return SandboxResult(
                success=False,
                stdout=e.stdout or "",
                stderr=e.stderr or "",
                error_type="timeout",
                return_code=-1,
                execution_time_sec=elapsed,
            )
    finally:
        # Always clean up the temp file
        try:
            os.unlink(tmp_path)
        except OSError:
            pass
