"""Multi-format tool call parser for extracting structured tool calls from model responses.

Parses tool calls from raw model response text in multiple formats:
1. Hermes format: <tool_call>{...}</tool_call>
2. JSON code blocks: ```json {...} ```
3. Raw JSON objects with 'name' or 'function' key

Priority chain: Hermes tags first, then JSON code blocks, then raw JSON.
Always strips <think>...</think> blocks before parsing.
Never raises exceptions — returns None on any failure.
"""

import json
import re
from dataclasses import dataclass
from typing import Optional

from mtb.quality_benchmarks.utils import _strip_thinking


@dataclass
class ToolCall:
    """A structured tool call extracted from model response text.

    Fields:
        name: The tool/function name to call.
        arguments: Dictionary of arguments to pass to the tool.
    """

    name: str
    arguments: dict


def parse_tool_calls(response: str) -> Optional[list[ToolCall]]:
    """Extract structured tool calls from a model response string.

    Attempts parsing in priority order:
    1. Hermes format (<tool_call>...</tool_call> tags)
    2. JSON code blocks (```json ... ```)
    3. Raw JSON objects with 'name' or 'function' key

    Args:
        response: Raw model response text, potentially including
                  <think> blocks, prose, and tool call structures.

    Returns:
        A list of ToolCall objects if tool calls are found, or None
        if no valid tool calls are detected. Never raises exceptions.
    """
    try:
        # Strip thinking blocks first
        cleaned = _strip_thinking(response)

        # Try each parsing strategy in priority order
        result = _parse_hermes(cleaned)
        if result:
            return result

        result = _parse_json_blocks(cleaned)
        if result:
            return result

        result = _parse_raw_json(cleaned)
        if result:
            return result

        return None
    except Exception:
        return None


def _parse_hermes(text: str) -> Optional[list[ToolCall]]:
    """Parse Hermes-format tool calls: <tool_call>{...}</tool_call>.

    Handles single and multiple tool_call blocks, with whitespace/newlines
    inside the tags.
    """
    try:
        pattern = re.compile(r"<tool_call>\s*(.*?)\s*</tool_call>", re.DOTALL)
        matches = pattern.findall(text)
        if not matches:
            return None

        results = []
        for match in matches:
            parsed = _try_parse_json(match.strip())
            if parsed is None:
                continue
            tool_calls = _extract_tool_calls_from_parsed(parsed)
            if tool_calls:
                results.extend(tool_calls)

        return results if results else None
    except Exception:
        return None


def _parse_json_blocks(text: str) -> Optional[list[ToolCall]]:
    """Parse tool calls from Markdown JSON code blocks.

    Handles ```json ... ``` blocks, including arrays of tool calls
    within a single block and multiple separate blocks.
    """
    try:
        pattern = re.compile(r"```(?:json)?\s*\n?(.*?)\n?\s*```", re.DOTALL)
        matches = pattern.findall(text)
        if not matches:
            return None

        results = []
        for match in matches:
            parsed = _try_parse_json(match.strip())
            if parsed is None:
                continue
            tool_calls = _extract_tool_calls_from_parsed(parsed)
            if tool_calls:
                results.extend(tool_calls)

        return results if results else None
    except Exception:
        return None


def _parse_raw_json(text: str) -> Optional[list[ToolCall]]:
    """Parse tool calls from raw JSON objects embedded in text.

    Finds JSON objects with 'name' or 'function' key in the response text.
    Ignores JSON objects that don't look like tool calls.
    """
    try:
        # Find all potential JSON objects in the text
        candidates = _find_json_objects(text)
        results = []
        for candidate in candidates:
            parsed = _try_parse_json(candidate)
            if parsed is None:
                continue
            tool_calls = _extract_tool_calls_from_parsed(parsed)
            if tool_calls:
                results.extend(tool_calls)

        return results if results else None
    except Exception:
        return None


def _find_json_objects(text: str) -> list[str]:
    """Find potential JSON object strings in text by matching braces/brackets.

    Returns a list of candidate JSON strings found in the text.
    """
    candidates = []
    i = 0
    while i < len(text):
        if text[i] in "{[":
            end = _find_matching_brace(text, i)
            if end is not None:
                candidate = text[i : end + 1]
                # Only consider if it's substantial enough to be a tool call
                if len(candidate) > 5:
                    candidates.append(candidate)
                i = end + 1
            else:
                i += 1
        else:
            i += 1
    return candidates


def _find_matching_brace(text: str, start: int) -> Optional[int]:
    """Find the matching closing brace/bracket for an opening one.

    Returns the index of the matching close, or None if not found.
    """
    open_char = text[start]
    close_char = "}" if open_char == "{" else "]"
    depth = 0
    in_string = False
    escape_next = False

    for i in range(start, len(text)):
        ch = text[i]
        if escape_next:
            escape_next = False
            continue
        if ch == "\\":
            if in_string:
                escape_next = True
            continue
        if ch == '"':
            in_string = not in_string
            continue
        if in_string:
            continue
        if ch == open_char:
            depth += 1
        elif ch == close_char:
            depth -= 1
            if depth == 0:
                return i
    return None


def _try_parse_json(text: str) -> object:
    """Try to parse JSON text, handling single quotes and other quirks.

    Returns the parsed object, or None if parsing fails.
    """
    if not text:
        return None

    # Try standard JSON first
    try:
        return json.loads(text)
    except (json.JSONDecodeError, ValueError):
        pass

    # Try fixing single quotes -> double quotes
    try:
        fixed = _fix_single_quotes(text)
        return json.loads(fixed)
    except (json.JSONDecodeError, ValueError):
        pass

    return None


def _fix_single_quotes(text: str) -> str:
    """Attempt to convert single-quoted JSON to double-quoted JSON.

    This is a best-effort transformation for model outputs that use
    Python-style single quotes instead of JSON double quotes.

    Escaped apostrophes (\\') inside values are preserved by temporarily
    replacing them with a placeholder before the quote swap.
    """
    # Preserve escaped apostrophes (e.g., it\'s) by replacing with placeholder
    _ESCAPED_APOS_PLACEHOLDER = "\x00ESCAPED_APOS\x00"
    text = text.replace("\\'", _ESCAPED_APOS_PLACEHOLDER)

    # Replace single quotes that are likely JSON delimiters
    # This is a simple heuristic: replace ' with " when it appears
    # at typical JSON boundaries
    result = []
    i = 0
    in_double_string = False
    while i < len(text):
        ch = text[i]
        if ch == '"' and (i == 0 or text[i - 1] != "\\"):
            in_double_string = not in_double_string
            result.append(ch)
        elif ch == "'" and not in_double_string:
            result.append('"')
        else:
            result.append(ch)
        i += 1
    fixed = "".join(result)

    # Restore escaped apostrophes
    fixed = fixed.replace(_ESCAPED_APOS_PLACEHOLDER, "'")
    return fixed


def _extract_tool_calls_from_parsed(parsed: object) -> list[ToolCall]:
    """Extract ToolCall objects from a parsed JSON structure.

    Handles:
    - Single dict with 'name' + 'arguments' keys
    - Single dict with 'function' key (maps to name)
    - Array of such dicts
    - Nested 'function' dict with 'name' and 'arguments' inside

    Returns empty list if the parsed object doesn't look like a tool call.
    """
    results = []

    if isinstance(parsed, list):
        for item in parsed:
            if isinstance(item, dict):
                tool_call = _dict_to_tool_call(item)
                if tool_call:
                    results.append(tool_call)
    elif isinstance(parsed, dict):
        tool_call = _dict_to_tool_call(parsed)
        if tool_call:
            results.append(tool_call)

    return results


def _dict_to_tool_call(d: dict) -> Optional[ToolCall]:
    """Convert a dictionary to a ToolCall if it has the right structure.

    Recognizes these patterns:
    1. {"name": "...", "arguments": {...}} — standard format
    2. {"function": "...", "arguments": {...}} — function key variant
    3. {"name": "...", "parameters": {...}} — parameters variant
    4. {"function": {"name": "...", "arguments": {...}}} — nested function
    5. {"name": "..."} — name only, empty arguments

    Returns None if the dict doesn't look like a tool call.
    """
    # Pattern 4: Nested function object
    if "function" in d and isinstance(d["function"], dict):
        inner = d["function"]
        name = inner.get("name")
        if name and isinstance(name, str):
            arguments = inner.get("arguments", {})
            if isinstance(arguments, str):
                # Sometimes arguments is a JSON string
                try:
                    arguments = json.loads(arguments)
                except (json.JSONDecodeError, ValueError):
                    arguments = {}
            if not isinstance(arguments, dict):
                arguments = {}
            return ToolCall(name=name, arguments=arguments)

    # Pattern 1: Standard name + arguments
    name = d.get("name") or d.get("tool_name") or d.get("tool")
    if name and isinstance(name, str):
        arguments = d.get("arguments") or d.get("parameters") or d.get("params") or {}
        if isinstance(arguments, str):
            try:
                arguments = json.loads(arguments)
            except (json.JSONDecodeError, ValueError):
                arguments = {}
        if not isinstance(arguments, dict):
            arguments = {}
        return ToolCall(name=name, arguments=arguments)

    # Pattern 2: function key as string
    func = d.get("function")
    if func and isinstance(func, str):
        arguments = d.get("arguments") or d.get("parameters") or d.get("params") or {}
        if isinstance(arguments, str):
            try:
                arguments = json.loads(arguments)
            except (json.JSONDecodeError, ValueError):
                arguments = {}
        if not isinstance(arguments, dict):
            arguments = {}
        return ToolCall(name=func, arguments=arguments)

    return None
