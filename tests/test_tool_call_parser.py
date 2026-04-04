"""Comprehensive tests for the tool call parser module.

Tests cover all VAL-PARSER assertions from the validation contract:
- VAL-PARSER-001: Hermes format — single tool call
- VAL-PARSER-002: Hermes format — multiple tool calls
- VAL-PARSER-003: Hermes format — with whitespace/newlines inside tags
- VAL-PARSER-004: JSON code block — single tool call
- VAL-PARSER-005: JSON code block — tool call with surrounding text
- VAL-PARSER-006: JSON code block — multiple code blocks
- VAL-PARSER-007: JSON code block — array of tool calls
- VAL-PARSER-008: Raw JSON — single object with "name" key
- VAL-PARSER-009: Raw JSON — embedded in prose
- VAL-PARSER-010: Raw JSON — ignores non-tool-call JSON
- VAL-PARSER-011: Think block stripping — all variants
- VAL-PARSER-014: No tool calls — pure text response
- VAL-PARSER-016: Malformed JSON — graceful handling
- VAL-PARSER-020: Mixed text + tool call response
- VAL-PARSER-023: Nested arguments — deeply nested dict
- VAL-PARSER-024: Arguments with array values
- VAL-PARSER-025: Empty arguments object
- VAL-PARSER-026: Return type contract
"""

import pytest

from mtb.quality_benchmarks.tool_call_parser import ToolCall, parse_tool_calls


# =============================================================================
# VAL-PARSER-001: Hermes format — single tool call
# =============================================================================
class TestHermesSingle:
    def test_basic_hermes_single(self):
        """Parser extracts a single tool call from Hermes-style XML tags."""
        response = (
            '<tool_call>{"name": "get_weather", '
            '"arguments": {"location": "San Francisco"}}</tool_call>'
        )
        result = parse_tool_calls(response)
        assert result is not None
        assert len(result) == 1
        assert result[0].name == "get_weather"
        assert result[0].arguments == {"location": "San Francisco"}

    def test_hermes_with_surrounding_text(self):
        """Hermes tags in the middle of other text."""
        response = (
            "I'll call the weather tool for you.\n"
            '<tool_call>{"name": "get_weather", '
            '"arguments": {"location": "NYC"}}</tool_call>\n'
            "That should give us the forecast."
        )
        result = parse_tool_calls(response)
        assert result is not None
        assert len(result) == 1
        assert result[0].name == "get_weather"
        assert result[0].arguments == {"location": "NYC"}


# =============================================================================
# VAL-PARSER-002: Hermes format — multiple tool calls
# =============================================================================
class TestHermesMultiple:
    def test_two_hermes_tool_calls(self):
        """Parser extracts multiple tool calls from several <tool_call> blocks."""
        response = (
            '<tool_call>{"name": "get_weather", '
            '"arguments": {"location": "SF"}}</tool_call>\n'
            '<tool_call>{"name": "get_time", '
            '"arguments": {"timezone": "PST"}}</tool_call>'
        )
        result = parse_tool_calls(response)
        assert result is not None
        assert len(result) == 2
        assert result[0].name == "get_weather"
        assert result[0].arguments == {"location": "SF"}
        assert result[1].name == "get_time"
        assert result[1].arguments == {"timezone": "PST"}


# =============================================================================
# VAL-PARSER-003: Hermes format — with whitespace/newlines inside tags
# =============================================================================
class TestHermesWhitespace:
    def test_whitespace_inside_tags(self):
        """Hermes tags with extra whitespace and newlines inside."""
        response = (
            "<tool_call>\n"
            "  {\n"
            '    "name": "get_weather",\n'
            '    "arguments": {\n'
            '      "location": "London"\n'
            "    }\n"
            "  }\n"
            "</tool_call>"
        )
        result = parse_tool_calls(response)
        assert result is not None
        assert len(result) == 1
        assert result[0].name == "get_weather"
        assert result[0].arguments == {"location": "London"}

    def test_leading_trailing_spaces(self):
        """Whitespace around JSON inside tags."""
        response = (
            '<tool_call>   {"name": "get_weather", '
            '"arguments": {"location": "Tokyo"}}   </tool_call>'
        )
        result = parse_tool_calls(response)
        assert result is not None
        assert len(result) == 1
        assert result[0].name == "get_weather"


# =============================================================================
# VAL-PARSER-004: JSON code block — single tool call
# =============================================================================
class TestJsonBlockSingle:
    def test_json_code_block_single(self):
        """Parser extracts a tool call from a Markdown JSON code block."""
        response = (
            "```json\n"
            '{"name": "get_weather", "arguments": {"location": "Paris"}}\n'
            "```"
        )
        result = parse_tool_calls(response)
        assert result is not None
        assert len(result) == 1
        assert result[0].name == "get_weather"
        assert result[0].arguments == {"location": "Paris"}

    def test_plain_code_block(self):
        """Code block without 'json' language specifier."""
        response = (
            "```\n"
            '{"name": "get_weather", "arguments": {"location": "Berlin"}}\n'
            "```"
        )
        result = parse_tool_calls(response)
        assert result is not None
        assert len(result) == 1
        assert result[0].name == "get_weather"


# =============================================================================
# VAL-PARSER-005: JSON code block — tool call with surrounding text
# =============================================================================
class TestJsonBlockSurroundingText:
    def test_code_block_with_surrounding_text(self):
        """Parser extracts tool call from code block embedded in text."""
        response = (
            "Sure, I'll check the weather for you.\n\n"
            "```json\n"
            '{"name": "get_weather", "arguments": {"location": "Miami"}}\n'
            "```\n\n"
            "This will return the current weather in Miami."
        )
        result = parse_tool_calls(response)
        assert result is not None
        assert len(result) == 1
        assert result[0].name == "get_weather"
        assert result[0].arguments == {"location": "Miami"}


# =============================================================================
# VAL-PARSER-006: JSON code block — multiple code blocks
# =============================================================================
class TestJsonBlockMultiple:
    def test_multiple_code_blocks(self):
        """Parser extracts tool calls from multiple JSON code blocks."""
        response = (
            "First call:\n"
            "```json\n"
            '{"name": "get_weather", "arguments": {"location": "SF"}}\n'
            "```\n"
            "Second call:\n"
            "```json\n"
            '{"name": "get_time", "arguments": {"timezone": "EST"}}\n'
            "```"
        )
        result = parse_tool_calls(response)
        assert result is not None
        assert len(result) == 2
        assert result[0].name == "get_weather"
        assert result[1].name == "get_time"


# =============================================================================
# VAL-PARSER-007: JSON code block — array of tool calls
# =============================================================================
class TestJsonBlockArray:
    def test_array_of_tool_calls(self):
        """Parser handles a single JSON code block containing an array of tool calls."""
        response = (
            "```json\n"
            "[\n"
            '  {"name": "get_weather", "arguments": {"location": "SF"}},\n'
            '  {"name": "get_time", "arguments": {"timezone": "PST"}}\n'
            "]\n"
            "```"
        )
        result = parse_tool_calls(response)
        assert result is not None
        assert len(result) == 2
        assert result[0].name == "get_weather"
        assert result[1].name == "get_time"


# =============================================================================
# VAL-PARSER-008: Raw JSON — single object with "name" key
# =============================================================================
class TestRawJsonSingle:
    def test_raw_json_single(self):
        """Parser extracts a tool call from raw JSON (no tags, no code block)."""
        response = '{"name": "get_weather", "arguments": {"location": "Chicago"}}'
        result = parse_tool_calls(response)
        assert result is not None
        assert len(result) == 1
        assert result[0].name == "get_weather"
        assert result[0].arguments == {"location": "Chicago"}


# =============================================================================
# VAL-PARSER-009: Raw JSON — embedded in prose
# =============================================================================
class TestRawJsonInProse:
    def test_raw_json_in_prose(self):
        """Parser extracts a tool call from raw JSON embedded in natural language."""
        response = (
            "I'll call the tool now: "
            '{"name": "get_weather", "arguments": {"location": "LA"}} '
            "to get the weather."
        )
        result = parse_tool_calls(response)
        assert result is not None
        assert len(result) >= 1
        assert result[0].name == "get_weather"
        assert result[0].arguments["location"] == "LA"


# =============================================================================
# VAL-PARSER-010: Raw JSON — ignores non-tool-call JSON
# =============================================================================
class TestRawJsonNonToolCall:
    def test_non_tool_call_json_ignored(self):
        """Parser does NOT treat arbitrary JSON without 'name' key as tool calls."""
        response = '{"temperature": 72, "humidity": 45, "wind": "5mph"}'
        result = parse_tool_calls(response)
        assert result is None or len(result) == 0

    def test_json_without_function_structure(self):
        """Parser ignores JSON that has no tool call structure."""
        response = '{"data": [1, 2, 3], "total": 6}'
        result = parse_tool_calls(response)
        assert result is None or len(result) == 0

    def test_json_with_name_but_not_tool_like(self):
        """JSON with 'name' key but with a string value is treated as a tool call.

        This is by design — any JSON with a 'name' key could be a tool call.
        The parser is permissive for name+arguments structures.
        """
        # A dict with 'name' and no arguments still parses (empty args)
        response = '{"name": "get_time"}'
        result = parse_tool_calls(response)
        # This has a 'name' key, so it IS treated as a tool call
        assert result is not None
        assert len(result) == 1
        assert result[0].name == "get_time"


# =============================================================================
# VAL-PARSER-011: Think block stripping — all variants
# =============================================================================
class TestThinkBlockStripping:
    def test_hermes_after_think_block(self):
        """Think blocks stripped before Hermes format parsing."""
        response = (
            "<think>Let me think about what tool to call...</think>\n"
            '<tool_call>{"name": "get_weather", '
            '"arguments": {"location": "Boston"}}</tool_call>'
        )
        result = parse_tool_calls(response)
        assert result is not None
        assert len(result) == 1
        assert result[0].name == "get_weather"
        assert result[0].arguments == {"location": "Boston"}

    def test_json_block_after_think_block(self):
        """Think blocks stripped before JSON code block parsing."""
        response = (
            "<think>I need to check the weather API...</think>\n"
            "```json\n"
            '{"name": "get_weather", "arguments": {"location": "Denver"}}\n'
            "```"
        )
        result = parse_tool_calls(response)
        assert result is not None
        assert len(result) == 1
        assert result[0].name == "get_weather"

    def test_unclosed_think_tag(self):
        """Unclosed <think> tag (model truncated mid-thinking) handled gracefully."""
        response = (
            "<think>I'm still thinking about which tool..."
            # No closing </think> tag - model was truncated
        )
        result = parse_tool_calls(response)
        # Should not crash, returns None since no tool call remains
        assert result is None or len(result) == 0

    def test_unclosed_think_with_tool_call_before(self):
        """Unclosed think that comes after a valid tool call in prose."""
        # The think stripping removes from <think> to end of string
        # So we test that unclosed think doesn't crash
        response = (
            '<tool_call>{"name": "get_weather", '
            '"arguments": {"location": "Seattle"}}</tool_call>\n'
            "<think>Processing the result..."
        )
        result = parse_tool_calls(response)
        assert result is not None
        assert len(result) == 1
        assert result[0].name == "get_weather"


# =============================================================================
# VAL-PARSER-014: No tool calls — pure text response
# =============================================================================
class TestNoToolCalls:
    def test_pure_text_response(self):
        """Response with no tool calls returns None or empty list."""
        response = "The weather in San Francisco is usually foggy and cool."
        result = parse_tool_calls(response)
        assert result is None or len(result) == 0

    def test_empty_string(self):
        """Empty string returns None."""
        result = parse_tool_calls("")
        assert result is None or len(result) == 0

    def test_whitespace_only(self):
        """Whitespace-only input returns None."""
        result = parse_tool_calls("   \n\t  ")
        assert result is None or len(result) == 0


# =============================================================================
# VAL-PARSER-016: Malformed JSON — graceful handling (consolidated)
# =============================================================================
class TestMalformedJson:
    def test_incomplete_json_in_hermes(self):
        """Incomplete JSON inside Hermes tags handled gracefully."""
        response = (
            '<tool_call>{"name": "get_weather", "arguments": {"location":</tool_call>'
        )
        result = parse_tool_calls(response)
        # Should not raise; returns None or empty
        assert result is None or isinstance(result, list)

    def test_truncated_response_mid_json(self):
        """Truncated response mid-JSON doesn't crash."""
        response = '{"name": "get_weather", "arguments": {"locat'
        result = parse_tool_calls(response)
        assert result is None or isinstance(result, list)

    def test_extra_text_inside_hermes_tags(self):
        """Extra text mixed with JSON inside Hermes tags."""
        response = (
            "<tool_call>Here is the tool call: "
            '{"name": "get_weather", "arguments": {"location": "SF"}}'
            "</tool_call>"
        )
        # The extra text before the JSON means json.loads will fail on the
        # full content, but we try to parse what we can
        result = parse_tool_calls(response)
        # May or may not parse depending on implementation
        assert result is None or isinstance(result, list)

    def test_single_quotes_json(self):
        """Single-quoted JSON (Python style) handled gracefully."""
        response = (
            "<tool_call>{'name': 'get_weather', "
            "'arguments': {'location': 'SF'}}</tool_call>"
        )
        result = parse_tool_calls(response)
        # Should attempt to fix single quotes and parse
        assert result is not None
        assert len(result) == 1
        assert result[0].name == "get_weather"
        assert result[0].arguments == {"location": "SF"}

    def test_completely_invalid_json(self):
        """Completely invalid content inside tags."""
        response = "<tool_call>not json at all</tool_call>"
        result = parse_tool_calls(response)
        assert result is None or isinstance(result, list)

    def test_empty_hermes_tags(self):
        """Empty content inside Hermes tags."""
        response = "<tool_call></tool_call>"
        result = parse_tool_calls(response)
        assert result is None or isinstance(result, list)

    def test_nested_malformed_does_not_raise(self):
        """Deeply nested malformed JSON doesn't crash."""
        response = '<tool_call>{{{{"name": bad}}}</tool_call>'
        result = parse_tool_calls(response)
        # Must not raise
        assert result is None or isinstance(result, list)


# =============================================================================
# VAL-PARSER-020: Mixed text + tool call response
# =============================================================================
class TestMixedTextToolCall:
    def test_text_then_tool_call_then_text(self):
        """Explanatory text, then a tool call, then more text — tool call extracted."""
        response = (
            "Let me check the weather for you.\n\n"
            '<tool_call>{"name": "get_weather", '
            '"arguments": {"location": "Austin"}}</tool_call>\n\n'
            "I've submitted the request. You should see results shortly."
        )
        result = parse_tool_calls(response)
        assert result is not None
        assert len(result) == 1
        assert result[0].name == "get_weather"
        assert result[0].arguments == {"location": "Austin"}


# =============================================================================
# VAL-PARSER-023: Nested arguments — deeply nested dict
# =============================================================================
class TestNestedArguments:
    def test_deeply_nested_dict(self):
        """Tool call with deeply nested argument objects parsed correctly."""
        response = (
            '<tool_call>{"name": "update_user", "arguments": '
            '{"user": {"name": "Alice", "address": {"city": "NYC", '
            '"zip": "10001"}}}}</tool_call>'
        )
        result = parse_tool_calls(response)
        assert result is not None
        assert len(result) == 1
        assert result[0].arguments["user"]["address"]["city"] == "NYC"
        assert result[0].arguments["user"]["address"]["zip"] == "10001"
        assert result[0].arguments["user"]["name"] == "Alice"


# =============================================================================
# VAL-PARSER-024: Arguments with array values
# =============================================================================
class TestArrayArguments:
    def test_array_in_arguments(self):
        """Tool call with array-valued arguments parsed correctly."""
        response = (
            '<tool_call>{"name": "send_email", "arguments": '
            '{"recipients": ["alice@example.com", "bob@example.com"], '
            '"subject": "Meeting"}}</tool_call>'
        )
        result = parse_tool_calls(response)
        assert result is not None
        assert len(result) == 1
        assert isinstance(result[0].arguments["recipients"], list)
        assert len(result[0].arguments["recipients"]) == 2
        assert result[0].arguments["recipients"][0] == "alice@example.com"
        assert result[0].arguments["recipients"][1] == "bob@example.com"


# =============================================================================
# VAL-PARSER-025: Empty arguments object
# =============================================================================
class TestEmptyArguments:
    def test_empty_arguments(self):
        """Tool call with empty arguments: {} parsed correctly."""
        response = '<tool_call>{"name": "get_time", "arguments": {}}</tool_call>'
        result = parse_tool_calls(response)
        assert result is not None
        assert len(result) == 1
        assert result[0].name == "get_time"
        assert result[0].arguments == {}

    def test_missing_arguments_key(self):
        """Tool call with no arguments key defaults to empty dict."""
        response = '<tool_call>{"name": "get_time"}</tool_call>'
        result = parse_tool_calls(response)
        assert result is not None
        assert len(result) == 1
        assert result[0].name == "get_time"
        assert result[0].arguments == {}


# =============================================================================
# VAL-PARSER-026: Return type contract
# =============================================================================
class TestReturnTypeContract:
    """parse_tool_calls returns list[ToolCall] or None, never raises."""

    @pytest.mark.parametrize(
        "input_text",
        [
            # Valid Hermes
            '<tool_call>{"name": "test", "arguments": {}}</tool_call>',
            # Valid JSON block
            '```json\n{"name": "test", "arguments": {}}\n```',
            # Valid raw JSON
            '{"name": "test", "arguments": {}}',
            # Pure text
            "Just a normal response with no tool calls.",
            # Empty string
            "",
            # Very long text
            "x" * 10000,
            # Binary-like text
            "\x00\x01\x02\x03" * 100,
            # Unicode text
            "こんにちは世界 🌍 مرحبا",
            # Malformed JSON
            '{"name": broken}',
            # HTML-like but not tool calls
            "<div>Hello</div><p>World</p>",
        ],
        ids=[
            "valid_hermes",
            "valid_json_block",
            "valid_raw_json",
            "pure_text",
            "empty_string",
            "very_long_text",
            "binary_like",
            "unicode",
            "malformed_json",
            "html_like",
        ],
    )
    def test_never_raises(self, input_text):
        """parse_tool_calls never raises on any input."""
        result = parse_tool_calls(input_text)
        assert result is None or isinstance(result, list)
        if isinstance(result, list):
            for item in result:
                assert isinstance(item, ToolCall)
                assert isinstance(item.name, str)
                assert isinstance(item.arguments, dict)


# =============================================================================
# Additional coverage: function key variant
# =============================================================================
class TestFunctionKeyVariant:
    def test_function_key_maps_to_name(self):
        """'function' key in JSON maps to ToolCall.name."""
        response = (
            '<tool_call>{"function": "get_weather", '
            '"arguments": {"location": "Portland"}}</tool_call>'
        )
        result = parse_tool_calls(response)
        assert result is not None
        assert len(result) == 1
        assert result[0].name == "get_weather"
        assert result[0].arguments == {"location": "Portland"}

    def test_nested_function_object(self):
        """Nested function object: {"function": {"name": ..., "arguments": ...}}."""
        response = (
            "```json\n"
            '{"function": {"name": "get_weather", '
            '"arguments": {"location": "Seattle"}}}\n'
            "```"
        )
        result = parse_tool_calls(response)
        assert result is not None
        assert len(result) == 1
        assert result[0].name == "get_weather"
        assert result[0].arguments == {"location": "Seattle"}


# =============================================================================
# Additional coverage: Priority chain
# =============================================================================
class TestPriorityChain:
    def test_hermes_takes_priority_over_json_block(self):
        """When both Hermes and JSON blocks are present, Hermes wins."""
        response = (
            '<tool_call>{"name": "hermes_tool", "arguments": {}}</tool_call>\n'
            "```json\n"
            '{"name": "json_block_tool", "arguments": {}}\n'
            "```"
        )
        result = parse_tool_calls(response)
        assert result is not None
        # Hermes should be parsed first and returned
        assert result[0].name == "hermes_tool"


# =============================================================================
# Additional coverage: ToolCall dataclass
# =============================================================================
class TestToolCallDataclass:
    def test_toolcall_fields(self):
        """ToolCall has name and arguments fields."""
        tc = ToolCall(name="test", arguments={"key": "value"})
        assert tc.name == "test"
        assert tc.arguments == {"key": "value"}

    def test_toolcall_equality(self):
        """ToolCall equality based on fields."""
        tc1 = ToolCall(name="test", arguments={"key": "value"})
        tc2 = ToolCall(name="test", arguments={"key": "value"})
        assert tc1 == tc2

    def test_toolcall_repr(self):
        """ToolCall has a useful repr."""
        tc = ToolCall(name="get_weather", arguments={"location": "SF"})
        repr_str = repr(tc)
        assert "get_weather" in repr_str
        assert "ToolCall" in repr_str


# =============================================================================
# Additional coverage: parameters key variant
# =============================================================================
class TestParametersKeyVariant:
    def test_parameters_key_accepted(self):
        """'parameters' key used instead of 'arguments'."""
        response = (
            '<tool_call>{"name": "search", '
            '"parameters": {"query": "test"}}</tool_call>'
        )
        result = parse_tool_calls(response)
        assert result is not None
        assert len(result) == 1
        assert result[0].name == "search"
        assert result[0].arguments == {"query": "test"}
