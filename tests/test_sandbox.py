"""Tests for the code execution sandbox module.

Covers: valid execution, timeout enforcement, error handling, temp file cleanup,
empty/whitespace code, stderr separation, markdown fence stripping, assertion
detection, concurrent calls, and SandboxResult dataclass fields.

Maps to validation contract assertions VAL-SANDBOX-001 through VAL-SANDBOX-030.
"""

import os
import tempfile
import time
from concurrent.futures import ThreadPoolExecutor
from dataclasses import fields
from typing import Optional

import pytest

from mtb.quality_benchmarks.sandbox import SandboxResult, execute_code


# ---------------------------------------------------------------------------
# VAL-SANDBOX-029: SandboxResult dataclass has all required fields
# ---------------------------------------------------------------------------
class TestSandboxResultDataclass:
    def test_required_fields_present(self):
        """SandboxResult must have: success, stdout, stderr, error_type, return_code, execution_time_sec."""
        field_names = {f.name for f in fields(SandboxResult)}
        expected = {
            "success",
            "stdout",
            "stderr",
            "error_type",
            "return_code",
            "execution_time_sec",
        }
        assert expected.issubset(
            field_names
        ), f"Missing fields: {expected - field_names}"

    def test_field_types_on_success(self):
        """Fields are correctly typed on a successful run."""
        result = execute_code('print("hello")')
        assert isinstance(result.success, bool)
        assert isinstance(result.stdout, str)
        assert isinstance(result.stderr, str)
        assert result.error_type is None or isinstance(result.error_type, str)
        assert isinstance(result.return_code, int)
        assert isinstance(result.execution_time_sec, float)

    def test_field_types_on_failure(self):
        """Fields are correctly typed on a failed run."""
        result = execute_code("1/0")
        assert isinstance(result.success, bool)
        assert isinstance(result.stdout, str)
        assert isinstance(result.stderr, str)
        assert result.error_type is None or isinstance(result.error_type, str)
        assert isinstance(result.return_code, int)
        assert isinstance(result.execution_time_sec, float)


# ---------------------------------------------------------------------------
# VAL-SANDBOX-001: Valid code produces captured stdout
# ---------------------------------------------------------------------------
class TestValidExecution:
    def test_hello_world(self):
        """print('hello world') returns captured stdout with success."""
        result = execute_code('print("hello world")')
        assert result.success is True
        assert result.stdout == "hello world\n"
        assert result.return_code == 0
        assert result.error_type is None

    def test_multiline_output(self):
        """Multi-line print output captured correctly."""
        code = 'print("line1")\nprint("line2")\nprint("line3")'
        result = execute_code(code)
        assert result.success is True
        assert result.stdout == "line1\nline2\nline3\n"

    def test_computation_result(self):
        """Computation + print produces correct output."""
        code = "x = 2 + 3\nprint(x)"
        result = execute_code(code)
        assert result.success is True
        assert result.stdout.strip() == "5"

    def test_stdlib_import_allowed(self):
        """Standard library imports work in sandbox."""
        code = "import math\nprint(math.pi)"
        result = execute_code(code)
        assert result.success is True
        assert "3.14159" in result.stdout

    def test_execution_time_tracked(self):
        """execution_time_sec is a positive float for valid code."""
        result = execute_code('print("fast")')
        assert result.execution_time_sec > 0.0
        assert result.execution_time_sec < 10.0  # shouldn't take long


# ---------------------------------------------------------------------------
# VAL-SANDBOX-004: Timeout enforcement — default 10 seconds
# ---------------------------------------------------------------------------
class TestTimeoutDefault:
    def test_default_timeout_kills_long_sleep(self):
        """Code sleeping 30s is killed within ~10s (default timeout)."""
        start = time.time()
        result = execute_code("import time; time.sleep(30)")
        elapsed = time.time() - start
        assert result.success is False
        assert result.error_type == "timeout"
        assert elapsed < 13  # 10s timeout + 3s overhead max


# ---------------------------------------------------------------------------
# VAL-SANDBOX-005: Timeout enforcement — custom value
# ---------------------------------------------------------------------------
class TestTimeoutCustom:
    def test_custom_timeout_3s(self):
        """timeout=3 kills time.sleep(10) within ~3s."""
        start = time.time()
        result = execute_code("import time; time.sleep(10)", timeout=3)
        elapsed = time.time() - start
        assert result.success is False
        assert result.error_type == "timeout"
        assert elapsed < 6  # 3s timeout + 3s overhead max

    def test_custom_timeout_1s(self):
        """timeout=1 kills time.sleep(10) quickly."""
        start = time.time()
        result = execute_code("import time; time.sleep(10)", timeout=1)
        elapsed = time.time() - start
        assert result.success is False
        assert result.error_type == "timeout"
        assert elapsed < 4


# ---------------------------------------------------------------------------
# VAL-SANDBOX-006: Infinite loop killed by timeout
# ---------------------------------------------------------------------------
class TestInfiniteLoop:
    def test_while_true_killed(self):
        """while True: pass is terminated by timeout."""
        result = execute_code("while True: pass", timeout=2)
        assert result.success is False
        assert result.error_type == "timeout"


# ---------------------------------------------------------------------------
# VAL-SANDBOX-008: Syntax error reported cleanly
# ---------------------------------------------------------------------------
class TestSyntaxError:
    def test_syntax_error_reported(self):
        """Invalid syntax reports SyntaxError cleanly."""
        result = execute_code("def foo(: pass")
        assert result.success is False
        assert result.error_type == "syntax_error"
        assert "SyntaxError" in result.stderr

    def test_syntax_error_no_sandbox_traceback(self):
        """Sandbox itself doesn't crash on syntax errors."""
        result = execute_code("if True\n  print('bad')")
        assert result.success is False
        assert result.error_type == "syntax_error"
        assert "SyntaxError" in result.stderr


# ---------------------------------------------------------------------------
# VAL-SANDBOX-009: Runtime errors reported cleanly
# (NameError, TypeError, ZeroDivisionError, ImportError)
# ---------------------------------------------------------------------------
class TestRuntimeErrors:
    @pytest.mark.parametrize(
        "code,error_class",
        [
            ("print(undefined_variable)", "NameError"),
            ('"hello" + 5', "TypeError"),
            ("1 / 0", "ZeroDivisionError"),
            ("import nonexistent_module_xyz", "ModuleNotFoundError"),
        ],
        ids=["NameError", "TypeError", "ZeroDivisionError", "ImportError"],
    )
    def test_runtime_error_reported(self, code, error_class):
        """Common runtime errors are reported cleanly without crashing sandbox."""
        result = execute_code(code)
        assert result.success is False
        assert error_class in result.stderr
        assert result.return_code != 0


# ---------------------------------------------------------------------------
# VAL-SANDBOX-014: Temp files cleaned up after execution
# (success, timeout, crash scenarios)
# ---------------------------------------------------------------------------
class TestTempFileCleanup:
    def _get_temp_files_before(self):
        """Get set of temp files before execution."""
        return set(os.listdir(tempfile.gettempdir()))

    def test_cleanup_after_success(self):
        """Temp files cleaned up after successful execution."""
        before = self._get_temp_files_before()
        execute_code('print("ok")')
        after = set(os.listdir(tempfile.gettempdir()))
        # No new sandbox temp files should remain
        new_files = after - before
        sandbox_files = [f for f in new_files if f.startswith("sandbox_")]
        assert len(sandbox_files) == 0, f"Leftover sandbox files: {sandbox_files}"

    def test_cleanup_after_timeout(self):
        """Temp files cleaned up after timeout."""
        before = self._get_temp_files_before()
        execute_code("import time; time.sleep(30)", timeout=1)
        after = set(os.listdir(tempfile.gettempdir()))
        new_files = after - before
        sandbox_files = [f for f in new_files if f.startswith("sandbox_")]
        assert len(sandbox_files) == 0, f"Leftover sandbox files: {sandbox_files}"

    def test_cleanup_after_crash(self):
        """Temp files cleaned up after code crash."""
        before = self._get_temp_files_before()
        execute_code("raise RuntimeError('crash')")
        after = set(os.listdir(tempfile.gettempdir()))
        new_files = after - before
        sandbox_files = [f for f in new_files if f.startswith("sandbox_")]
        assert len(sandbox_files) == 0, f"Leftover sandbox files: {sandbox_files}"


# ---------------------------------------------------------------------------
# VAL-SANDBOX-017: Empty code string handled
# ---------------------------------------------------------------------------
class TestEmptyCode:
    def test_empty_string(self):
        """Empty string doesn't crash; returns a defined result."""
        result = execute_code("")
        assert isinstance(result, SandboxResult)
        # Either success with empty stdout, or a graceful error
        assert result.stdout == "" or result.success is True

    def test_none_like_empty(self):
        """Passing whitespace-stripped empty string works."""
        result = execute_code("   ")
        assert isinstance(result, SandboxResult)


# ---------------------------------------------------------------------------
# VAL-SANDBOX-018: Whitespace-only code handled
# ---------------------------------------------------------------------------
class TestWhitespaceCode:
    def test_spaces_only(self):
        """Code with only spaces succeeds with empty stdout."""
        result = execute_code("   ")
        assert result.success is True
        assert result.stdout == ""

    def test_newlines_only(self):
        """Code with only newlines succeeds with empty stdout."""
        result = execute_code("\n\n\n")
        assert result.success is True
        assert result.stdout == ""

    def test_mixed_whitespace(self):
        """Code with mixed whitespace (spaces, tabs, newlines) succeeds."""
        result = execute_code("  \t  \n  \n  ")
        assert result.success is True
        assert result.stdout == ""


# ---------------------------------------------------------------------------
# VAL-SANDBOX-020: Stderr captured separately from stdout
# ---------------------------------------------------------------------------
class TestStderrSeparation:
    def test_stdout_and_stderr_separate(self):
        """stdout and stderr are captured in separate fields."""
        code = 'import sys; print("out"); print("err", file=sys.stderr)'
        result = execute_code(code)
        assert "out" in result.stdout
        assert "err" in result.stderr

    def test_only_stderr(self):
        """Code that only writes to stderr has empty stdout."""
        code = 'import sys; print("error_only", file=sys.stderr); sys.exit(1)'
        result = execute_code(code)
        assert "error_only" in result.stderr
        assert result.stdout == ""


# ---------------------------------------------------------------------------
# VAL-SANDBOX-023: Code with test assertions — all pass → success
# ---------------------------------------------------------------------------
class TestAssertionPass:
    def test_all_assertions_pass(self):
        """When all assertions pass, sandbox reports success."""
        code = """
def add(a, b):
    return a + b

assert add(2, 3) == 5
assert add(0, 0) == 0
assert add(-1, 1) == 0
"""
        result = execute_code(code)
        assert result.success is True

    def test_assertion_with_message(self):
        """Assertions with messages pass correctly."""
        code = """
x = 42
assert x == 42, "x should be 42"
"""
        result = execute_code(code)
        assert result.success is True


# ---------------------------------------------------------------------------
# VAL-SANDBOX-024: Code with test assertions — one fails → failure
# ---------------------------------------------------------------------------
class TestAssertionFail:
    def test_failing_assertion(self):
        """When an assertion fails, sandbox reports failure with AssertionError."""
        code = """
def add(a, b):
    return a + b

assert add(2, 3) == 6
"""
        result = execute_code(code)
        assert result.success is False
        assert "AssertionError" in result.stderr or "AssertionError" in result.stderr

    def test_assertion_error_in_stderr(self):
        """AssertionError appears in stderr."""
        code = "assert False, 'deliberate fail'"
        result = execute_code(code)
        assert result.success is False
        assert "AssertionError" in result.stderr


# ---------------------------------------------------------------------------
# VAL-SANDBOX-027: Code with markdown fences stripped
# ---------------------------------------------------------------------------
class TestMarkdownFenceStripping:
    def test_python_fences_stripped(self):
        """Code wrapped in ```python ... ``` fences executes correctly."""
        code = '```python\nprint("fenced")\n```'
        result = execute_code(code)
        assert result.success is True
        assert "fenced" in result.stdout

    def test_plain_fences_stripped(self):
        """Code wrapped in ``` ... ``` fences (no language) executes correctly."""
        code = '```\nprint("plain_fenced")\n```'
        result = execute_code(code)
        assert result.success is True
        assert "plain_fenced" in result.stdout

    def test_no_fences_unchanged(self):
        """Code without fences executes as-is."""
        code = 'print("no_fences")'
        result = execute_code(code)
        assert result.success is True
        assert "no_fences" in result.stdout

    def test_mixed_fences_and_text(self):
        """If code has explanation text outside fences, the fenced code is extracted."""
        code = (
            'Here is the code:\n```python\nprint("extracted")\n```\nThat should work!'
        )
        result = execute_code(code)
        assert result.success is True
        assert "extracted" in result.stdout

    def test_multiple_fence_blocks(self):
        """Multiple fenced blocks: content is extracted and concatenated."""
        code = "```python\nx = 1\n```\nSome text\n```python\nprint(x)\n```"
        result = execute_code(code)
        assert result.success is True
        assert "1" in result.stdout


# ---------------------------------------------------------------------------
# VAL-SANDBOX-030: Concurrent sandbox calls don't interfere
# ---------------------------------------------------------------------------
class TestConcurrentCalls:
    def test_parallel_executions_isolated(self):
        """Two sandbox calls running in parallel produce correct independent results."""
        code_a = 'import time; time.sleep(0.1); print("result_a")'
        code_b = 'import time; time.sleep(0.1); print("result_b")'

        with ThreadPoolExecutor(max_workers=2) as executor:
            future_a = executor.submit(execute_code, code_a)
            future_b = executor.submit(execute_code, code_b)
            result_a = future_a.result()
            result_b = future_b.result()

        assert result_a.success is True
        assert "result_a" in result_a.stdout
        assert "result_b" not in result_a.stdout

        assert result_b.success is True
        assert "result_b" in result_b.stdout
        assert "result_a" not in result_b.stdout

    def test_many_parallel_executions(self):
        """5 concurrent sandbox calls all produce correct results."""
        codes = [f'print("output_{i}")' for i in range(5)]

        with ThreadPoolExecutor(max_workers=5) as executor:
            results = list(executor.map(execute_code, codes))

        for i, result in enumerate(results):
            assert result.success is True
            assert f"output_{i}" in result.stdout

    def test_concurrent_with_failure(self):
        """Concurrent calls where one fails don't affect the other."""
        code_good = 'print("good")'
        code_bad = "1/0"

        with ThreadPoolExecutor(max_workers=2) as executor:
            future_good = executor.submit(execute_code, code_good)
            future_bad = executor.submit(execute_code, code_bad)
            result_good = future_good.result()
            result_bad = future_bad.result()

        assert result_good.success is True
        assert "good" in result_good.stdout

        assert result_bad.success is False
        assert "ZeroDivisionError" in result_bad.stderr
