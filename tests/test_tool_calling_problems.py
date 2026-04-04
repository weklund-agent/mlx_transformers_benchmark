"""Tests for tool calling problems: Tool Selection and Argument Accuracy.

Each subcategory has pass tests (correct responses) and fail tests (incorrect
responses), plus think-block-wrapped response tests.
"""

import json

import pytest

from mtb.quality_benchmarks.tool_calling_problems import (
    ARGUMENT_ACCURACY_PROBLEMS,
    TOOL_CALLING_NEW_PROBLEMS,
    TOOL_SELECTION_PROBLEMS,
    _check_aa_all_required_args,
    _check_aa_boolean_natural_lang,
    _check_aa_date_iso_format,
    _check_aa_email_extraction,
    _check_aa_enum_values,
    _check_aa_nested_objects,
    _check_aa_numeric_coercion,
    _check_aa_preserve_exact_strings,
    _check_ts_ambiguous_request,
    _check_ts_correct_tool,
    _check_ts_multiple_valid,
    _check_ts_nested_descriptions,
    _check_ts_none_selection,
    _check_ts_parameter_based,
    _check_ts_similar_names,
    _check_ts_specialized_vs_general,
)


# =============================================================================
# Structural tests
# =============================================================================


class TestToolCallingProblemsStructure:
    """Structural validation for the 16 new tool calling problems."""

    def test_total_problem_count(self):
        assert len(TOOL_CALLING_NEW_PROBLEMS) == 16

    def test_selection_count(self):
        assert len(TOOL_SELECTION_PROBLEMS) == 8

    def test_accuracy_count(self):
        assert len(ARGUMENT_ACCURACY_PROBLEMS) == 8

    def test_all_category_is_tool_calling(self):
        for p in TOOL_CALLING_NEW_PROBLEMS:
            assert p.category == "tool_calling", f"{p.name} has category {p.category}"

    def test_all_names_unique(self):
        names = [p.name for p in TOOL_CALLING_NEW_PROBLEMS]
        assert len(names) == len(set(names)), f"Duplicate names: {names}"

    def test_all_checks_are_callable(self):
        for p in TOOL_CALLING_NEW_PROBLEMS:
            assert callable(p.check), f"{p.name} check is not callable"

    def test_all_have_nonempty_prompts(self):
        for p in TOOL_CALLING_NEW_PROBLEMS:
            assert len(p.prompt) > 0, f"{p.name} has empty prompt"

    def test_all_max_tokens_at_least_256(self):
        for p in TOOL_CALLING_NEW_PROBLEMS:
            assert (
                p.max_tokens >= 256
            ), f"{p.name} has max_tokens={p.max_tokens}, need >= 256"

    def test_all_prompts_contain_tool_definitions(self):
        """Every prompt must contain at least one tool definition."""
        tool_keywords = [
            "function",
            "tool",
            "Tool:",
            '"name"',
            "parameters",
            "Description:",
        ]
        for p in TOOL_CALLING_NEW_PROBLEMS:
            prompt_lower = p.prompt.lower()
            has_tool_def = any(kw.lower() in prompt_lower for kw in tool_keywords)
            assert has_tool_def, f"{p.name} prompt does not contain tool definitions"

    def test_selection_names_have_ts_prefix(self):
        for p in TOOL_SELECTION_PROBLEMS:
            assert p.name.startswith("ts_"), f"{p.name} should start with 'ts_'"

    def test_accuracy_names_have_aa_prefix(self):
        for p in ARGUMENT_ACCURACY_PROBLEMS:
            assert p.name.startswith("aa_"), f"{p.name} should start with 'aa_'"

    def test_no_overlap_with_existing_names(self):
        """New problems must not share names with existing easy/hard/expert problems."""
        from mtb.quality_benchmarks.eval_problems import (
            EVAL_PROBLEMS,
            EXPERT_EVAL_PROBLEMS,
            HARD_EVAL_PROBLEMS,
        )

        existing_names = {
            p.name for p in EVAL_PROBLEMS + HARD_EVAL_PROBLEMS + EXPERT_EVAL_PROBLEMS
        }
        new_names = {p.name for p in TOOL_CALLING_NEW_PROBLEMS}
        overlap = existing_names & new_names
        assert len(overlap) == 0, f"Overlapping names with other tiers: {overlap}"


# =============================================================================
# Tool Selection – pass cases (VAL-TOOLCALL-100)
# =============================================================================


class TestToolSelectionPass:
    """Correct responses for all 8 tool selection problems."""

    def test_ts_correct_tool_json(self):
        response = json.dumps(
            {
                "name": "search_flights",
                "arguments": {
                    "origin": "New York",
                    "destination": "London",
                    "date": "2024-03-15",
                },
            }
        )
        assert _check_ts_correct_tool(response) is True

    def test_ts_correct_tool_hermes(self):
        response = (
            '<tool_call>{"name": "search_flights", "arguments": '
            '{"origin": "New York", "destination": "London", '
            '"date": "2024-03-15"}}</tool_call>'
        )
        assert _check_ts_correct_tool(response) is True

    def test_ts_ambiguous_request_pass(self):
        response = json.dumps(
            {
                "name": "schedule_meeting",
                "arguments": {
                    "title": "Call with Marketing",
                    "participants": ["marketing team"],
                    "date": "next Tuesday",
                    "time": "2:00 PM",
                },
            }
        )
        assert _check_ts_ambiguous_request(response) is True

    def test_ts_none_selection_pass(self):
        """Direct greeting without any tool calls."""
        response = (
            "Hello! I'm doing well, thank you for asking. How can I help you today?"
        )
        assert _check_ts_none_selection(response) is True

    def test_ts_similar_names_pass(self):
        response = json.dumps(
            {
                "name": "get_user_profile",
                "arguments": {"user_id": "alice_42"},
            }
        )
        assert _check_ts_similar_names(response) is True

    def test_ts_parameter_based_pass(self):
        response = json.dumps(
            {
                "name": "resize_image",
                "arguments": {
                    "file_path": "logo.png",
                    "width": 200,
                    "height": 100,
                },
            }
        )
        assert _check_ts_parameter_based(response) is True

    def test_ts_nested_descriptions_pass(self):
        response = json.dumps(
            {
                "name": "execute_query",
                "arguments": {
                    "query": "SELECT COUNT(*) FROM users WHERE created_at > '2024-01-01'",
                    "database": "analytics",
                },
            }
        )
        assert _check_ts_nested_descriptions(response) is True

    def test_ts_specialized_vs_general_pass(self):
        response = json.dumps(
            {
                "name": "translate_document",
                "arguments": {
                    "file_path": "report.pdf",
                    "source_lang": "de",
                    "target_lang": "en",
                    "preserve_formatting": True,
                },
            }
        )
        assert _check_ts_specialized_vs_general(response) is True

    def test_ts_multiple_valid_pass(self):
        response = json.dumps(
            {
                "name": "create_ticket",
                "arguments": {
                    "title": "500 error in checkout flow",
                    "description": "Customers getting 500 error on Pay Now",
                    "priority": "high",
                    "assignee": "backend-team",
                },
            }
        )
        assert _check_ts_multiple_valid(response) is True


# =============================================================================
# Tool Selection – fail cases (VAL-TOOLCALL-108)
# =============================================================================


class TestToolSelectionFail:
    """Wrong responses for all 8 tool selection problems."""

    def test_ts_correct_tool_wrong_tool(self):
        response = json.dumps(
            {
                "name": "book_hotel",
                "arguments": {"city": "London", "check_in": "2024-03-15"},
            }
        )
        assert _check_ts_correct_tool(response) is False

    def test_ts_correct_tool_no_tool(self):
        response = "I can help you find flights from New York to London."
        assert _check_ts_correct_tool(response) is False

    def test_ts_ambiguous_request_wrong_tool(self):
        response = json.dumps(
            {
                "name": "set_reminder",
                "arguments": {
                    "message": "Call marketing team",
                    "datetime": "next Tuesday 2pm",
                },
            }
        )
        assert _check_ts_ambiguous_request(response) is False

    def test_ts_none_selection_calls_tool(self):
        """Calling a tool for a simple greeting is wrong."""
        response = json.dumps(
            {
                "name": "search_database",
                "arguments": {"query": "greeting"},
            }
        )
        assert _check_ts_none_selection(response) is False

    def test_ts_similar_names_wrong_similar_tool(self):
        response = json.dumps(
            {
                "name": "get_user_preferences",
                "arguments": {"user_id": "alice_42"},
            }
        )
        assert _check_ts_similar_names(response) is False

    def test_ts_similar_names_wrong_activity(self):
        response = json.dumps(
            {
                "name": "get_user_activity",
                "arguments": {"user_id": "alice_42", "days": 7},
            }
        )
        assert _check_ts_similar_names(response) is False

    def test_ts_parameter_based_wrong_tool(self):
        response = json.dumps(
            {
                "name": "compress_image",
                "arguments": {"file_path": "logo.png", "quality": 80},
            }
        )
        assert _check_ts_parameter_based(response) is False

    def test_ts_nested_descriptions_wrong_tool(self):
        response = json.dumps(
            {
                "name": "list_tables",
                "arguments": {"database": "analytics"},
            }
        )
        assert _check_ts_nested_descriptions(response) is False

    def test_ts_specialized_vs_general_picks_general(self):
        response = json.dumps(
            {
                "name": "translate_text",
                "arguments": {
                    "text": "report content...",
                    "source_lang": "de",
                    "target_lang": "en",
                },
            }
        )
        assert _check_ts_specialized_vs_general(response) is False

    def test_ts_multiple_valid_picks_email(self):
        response = json.dumps(
            {
                "name": "send_email",
                "arguments": {
                    "to": "backend@company.com",
                    "subject": "Bug Report",
                    "body": "Checkout 500 error",
                },
            }
        )
        assert _check_ts_multiple_valid(response) is False

    def test_ts_multiple_valid_picks_message(self):
        response = json.dumps(
            {
                "name": "send_message",
                "arguments": {
                    "channel": "#bugs",
                    "message": "Checkout 500 error",
                },
            }
        )
        assert _check_ts_multiple_valid(response) is False


# =============================================================================
# Argument Accuracy – pass cases (VAL-TOOLCALL-200)
# =============================================================================


class TestArgumentAccuracyPass:
    """Correct argument formatting for all 8 argument accuracy problems."""

    def test_aa_date_iso_format_pass(self):
        response = json.dumps(
            {
                "name": "schedule_event",
                "arguments": {
                    "title": "Team Standup",
                    "date": "2024-06-15",
                    "time": "09:30",
                    "duration_minutes": 30,
                },
            }
        )
        assert _check_aa_date_iso_format(response) is True

    def test_aa_email_extraction_pass(self):
        response = json.dumps(
            {
                "name": "send_invitation",
                "arguments": {
                    "event_title": "Q2 Planning",
                    "recipients": ["alice@company.com", "bob@company.com"],
                    "message": "Please review the agenda beforehand.",
                },
            }
        )
        assert _check_aa_email_extraction(response) is True

    def test_aa_enum_values_pass(self):
        response = json.dumps(
            {
                "name": "create_alert",
                "arguments": {
                    "service_name": "payment-service",
                    "metric": "cpu_usage",
                    "threshold": 90,
                    "severity": "high",
                },
            }
        )
        assert _check_aa_enum_values(response) is True

    def test_aa_nested_objects_pass(self):
        response = json.dumps(
            {
                "name": "create_order",
                "arguments": {
                    "product_id": "SKU-7890",
                    "quantity": 3,
                    "shipping_address": {
                        "street": "456 Pine Ave",
                        "city": "Seattle",
                        "state": "WA",
                        "zip": "98101",
                        "country": "United States",
                    },
                },
            }
        )
        assert _check_aa_nested_objects(response) is True

    def test_aa_numeric_coercion_pass(self):
        response = json.dumps(
            {
                "name": "transfer_funds",
                "arguments": {
                    "from_account": "ACC-001",
                    "to_account": "ACC-002",
                    "amount": 1500,
                    "currency": "USD",
                },
            }
        )
        assert _check_aa_numeric_coercion(response) is True

    def test_aa_numeric_coercion_pass_float(self):
        response = json.dumps(
            {
                "name": "transfer_funds",
                "arguments": {
                    "from_account": "ACC-001",
                    "to_account": "ACC-002",
                    "amount": 1500.00,
                    "currency": "USD",
                },
            }
        )
        assert _check_aa_numeric_coercion(response) is True

    def test_aa_boolean_natural_lang_pass(self):
        response = json.dumps(
            {
                "name": "update_notification_settings",
                "arguments": {
                    "user_id": "usr_123",
                    "email_notifications": True,
                    "sms_notifications": False,
                },
            }
        )
        assert _check_aa_boolean_natural_lang(response) is True

    def test_aa_all_required_args_pass(self):
        response = json.dumps(
            {
                "name": "deploy_service",
                "arguments": {
                    "service_name": "auth-service",
                    "version": "2.1.0",
                    "environment": "staging",
                    "replicas": 3,
                },
            }
        )
        assert _check_aa_all_required_args(response) is True

    def test_aa_preserve_exact_strings_pass(self):
        response = json.dumps(
            {
                "name": "execute_command",
                "arguments": {"command": "grep -rn 'TODO' src/"},
            }
        )
        assert _check_aa_preserve_exact_strings(response) is True

    def test_aa_preserve_exact_strings_pass_with_dir(self):
        response = json.dumps(
            {
                "name": "execute_command",
                "arguments": {
                    "command": "grep -rn 'TODO' src/",
                    "working_directory": "/project",
                },
            }
        )
        assert _check_aa_preserve_exact_strings(response) is True


# =============================================================================
# Argument Accuracy – fail cases (VAL-TOOLCALL-208)
# =============================================================================


class TestArgumentAccuracyFail:
    """Wrong argument formatting for all 8 argument accuracy problems."""

    def test_aa_date_iso_wrong_format(self):
        """Date in wrong format (MM/DD/YYYY instead of YYYY-MM-DD)."""
        response = json.dumps(
            {
                "name": "schedule_event",
                "arguments": {
                    "title": "Team Standup",
                    "date": "06/15/2024",
                    "time": "09:30",
                },
            }
        )
        assert _check_aa_date_iso_format(response) is False

    def test_aa_date_iso_wrong_date(self):
        """Correct format but wrong date value."""
        response = json.dumps(
            {
                "name": "schedule_event",
                "arguments": {
                    "title": "Team Standup",
                    "date": "2024-06-16",
                    "time": "09:30",
                },
            }
        )
        assert _check_aa_date_iso_format(response) is False

    def test_aa_date_iso_no_tool_call(self):
        response = "Sure, I'll schedule a standup for June 15th at 9:30 AM."
        assert _check_aa_date_iso_format(response) is False

    def test_aa_email_extraction_missing_email(self):
        """Only one of two required emails."""
        response = json.dumps(
            {
                "name": "send_invitation",
                "arguments": {
                    "event_title": "Q2 Planning",
                    "recipients": ["alice@company.com"],
                },
            }
        )
        assert _check_aa_email_extraction(response) is False

    def test_aa_email_extraction_no_emails(self):
        """Names instead of email addresses."""
        response = json.dumps(
            {
                "name": "send_invitation",
                "arguments": {
                    "event_title": "Q2 Planning",
                    "recipients": ["Alice", "Bob"],
                },
            }
        )
        assert _check_aa_email_extraction(response) is False

    def test_aa_enum_values_wrong_enum(self):
        """Using 'urgent' instead of valid enum 'critical'."""
        response = json.dumps(
            {
                "name": "create_alert",
                "arguments": {
                    "service_name": "payment-service",
                    "metric": "cpu_usage",
                    "threshold": 90,
                    "severity": "urgent",
                },
            }
        )
        assert _check_aa_enum_values(response) is False

    def test_aa_enum_values_wrong_severity(self):
        """Wrong severity level entirely."""
        response = json.dumps(
            {
                "name": "create_alert",
                "arguments": {
                    "service_name": "payment-service",
                    "metric": "cpu_usage",
                    "threshold": 90,
                    "severity": "medium",
                },
            }
        )
        assert _check_aa_enum_values(response) is False

    def test_aa_nested_objects_flat_address(self):
        """Address fields at top level instead of nested."""
        response = json.dumps(
            {
                "name": "create_order",
                "arguments": {
                    "product_id": "SKU-7890",
                    "quantity": 3,
                    "city": "Seattle",
                    "state": "WA",
                    "zip": "98101",
                },
            }
        )
        assert _check_aa_nested_objects(response) is False

    def test_aa_nested_objects_wrong_city(self):
        """Nested but wrong city."""
        response = json.dumps(
            {
                "name": "create_order",
                "arguments": {
                    "product_id": "SKU-7890",
                    "quantity": 3,
                    "shipping_address": {
                        "street": "456 Pine Ave",
                        "city": "Portland",
                        "state": "OR",
                        "zip": "97201",
                        "country": "United States",
                    },
                },
            }
        )
        assert _check_aa_nested_objects(response) is False

    def test_aa_numeric_coercion_string_amount(self):
        """Amount as string instead of number."""
        response = (
            '{"name": "transfer_funds", "arguments": '
            '{"from_account": "ACC-001", "to_account": "ACC-002", '
            '"amount": "1500", "currency": "USD"}}'
        )
        assert _check_aa_numeric_coercion(response) is False

    def test_aa_numeric_coercion_wrong_amount(self):
        """Wrong amount value."""
        response = json.dumps(
            {
                "name": "transfer_funds",
                "arguments": {
                    "from_account": "ACC-001",
                    "to_account": "ACC-002",
                    "amount": 150,
                    "currency": "USD",
                },
            }
        )
        assert _check_aa_numeric_coercion(response) is False

    def test_aa_boolean_natural_lang_strings(self):
        """Boolean values as strings instead of actual booleans."""
        response = (
            '{"name": "update_notification_settings", "arguments": '
            '{"user_id": "usr_123", "email_notifications": "true", '
            '"sms_notifications": "false"}}'
        )
        assert _check_aa_boolean_natural_lang(response) is False

    def test_aa_boolean_natural_lang_inverted(self):
        """Boolean values inverted (email off, sms on)."""
        response = json.dumps(
            {
                "name": "update_notification_settings",
                "arguments": {
                    "user_id": "usr_123",
                    "email_notifications": False,
                    "sms_notifications": True,
                },
            }
        )
        assert _check_aa_boolean_natural_lang(response) is False

    def test_aa_all_required_args_missing_replicas(self):
        """Missing required 'replicas' argument."""
        response = json.dumps(
            {
                "name": "deploy_service",
                "arguments": {
                    "service_name": "auth-service",
                    "version": "2.1.0",
                    "environment": "staging",
                },
            }
        )
        assert _check_aa_all_required_args(response) is False

    def test_aa_all_required_args_wrong_service(self):
        """Wrong service name."""
        response = json.dumps(
            {
                "name": "deploy_service",
                "arguments": {
                    "service_name": "payment-service",
                    "version": "2.1.0",
                    "environment": "staging",
                    "replicas": 3,
                },
            }
        )
        assert _check_aa_all_required_args(response) is False

    def test_aa_preserve_exact_strings_modified_command(self):
        """Command was interpreted/expanded instead of preserved."""
        response = json.dumps(
            {
                "name": "execute_command",
                "arguments": {"command": "find src/ -type f -exec grep -l TODO {} \\;"},
            }
        )
        assert _check_aa_preserve_exact_strings(response) is False

    def test_aa_preserve_exact_strings_no_tool(self):
        response = "Sure, I'll run grep -rn 'TODO' src/ for you."
        assert _check_aa_preserve_exact_strings(response) is False


# =============================================================================
# Think block tests — all 16 problems with <think> wrapped responses
# (VAL-TOOLCALL-010)
# =============================================================================


class TestToolSelectionWithThinkBlocks:
    """Think-block-wrapped responses should still pass checks."""

    def test_ts_correct_tool_with_think(self):
        response = (
            "<think>The user wants to find flights. search_flights is correct.</think>\n"
            '{"name": "search_flights", "arguments": '
            '{"origin": "New York", "destination": "London", "date": "2024-03-15"}}'
        )
        assert _check_ts_correct_tool(response) is True

    def test_ts_ambiguous_request_with_think(self):
        response = (
            "<think>Setting up a call = scheduling a meeting, not a reminder.</think>\n"
            '{"name": "schedule_meeting", "arguments": '
            '{"title": "Marketing Call", "participants": ["marketing"], '
            '"date": "Tuesday", "time": "14:00"}}'
        )
        assert _check_ts_ambiguous_request(response) is True

    def test_ts_none_selection_with_think(self):
        response = (
            "<think>The user is just saying hello. No tool needed.</think>\n"
            "Hello! How can I help you today?"
        )
        assert _check_ts_none_selection(response) is True

    def test_ts_similar_names_with_think(self):
        response = (
            "<think>Need profile info (name, email), not preferences or activity.</think>\n"
            '{"name": "get_user_profile", "arguments": {"user_id": "alice_42"}}'
        )
        assert _check_ts_similar_names(response) is True

    def test_ts_parameter_based_with_think(self):
        response = (
            "<think>The user wants specific width/height. That's resize, not compress.</think>\n"
            '{"name": "resize_image", "arguments": '
            '{"file_path": "logo.png", "width": 200, "height": 100}}'
        )
        assert _check_ts_parameter_based(response) is True

    def test_ts_nested_descriptions_with_think(self):
        response = (
            "<think>This is a query to run, not list tables or describe.</think>\n"
            '{"name": "execute_query", "arguments": '
            '{"query": "SELECT COUNT(*) FROM users WHERE created_at > \'2024-01-01\'", '
            '"database": "analytics"}}'
        )
        assert _check_ts_nested_descriptions(response) is True

    def test_ts_specialized_vs_general_with_think(self):
        response = (
            "<think>50-page PDF requires translate_document, not translate_text.</think>\n"
            '{"name": "translate_document", "arguments": '
            '{"file_path": "report.pdf", "source_lang": "de", '
            '"target_lang": "en", "preserve_formatting": true}}'
        )
        assert _check_ts_specialized_vs_general(response) is True

    def test_ts_multiple_valid_with_think(self):
        response = (
            "<think>A bug should be logged as a ticket, not emailed.</think>\n"
            '{"name": "create_ticket", "arguments": '
            '{"title": "Checkout 500 error", "description": "Pay Now button", '
            '"priority": "high", "assignee": "backend-team"}}'
        )
        assert _check_ts_multiple_valid(response) is True


class TestArgumentAccuracyWithThinkBlocks:
    """Think-block-wrapped responses should still pass checks."""

    def test_aa_date_iso_with_think(self):
        response = (
            "<think>June 15th 2024 in ISO format is 2024-06-15.</think>\n"
            '{"name": "schedule_event", "arguments": '
            '{"title": "Team Standup", "date": "2024-06-15", '
            '"time": "09:30", "duration_minutes": 30}}'
        )
        assert _check_aa_date_iso_format(response) is True

    def test_aa_email_extraction_with_think(self):
        response = (
            "<think>I need to extract both email addresses.</think>\n"
            '{"name": "send_invitation", "arguments": '
            '{"event_title": "Q2 Planning", '
            '"recipients": ["alice@company.com", "bob@company.com"], '
            '"message": "Please review the agenda beforehand."}}'
        )
        assert _check_aa_email_extraction(response) is True

    def test_aa_enum_values_with_think(self):
        response = (
            "<think>The valid enums are low, medium, high, critical. User wants high.</think>\n"
            '{"name": "create_alert", "arguments": '
            '{"service_name": "payment-service", "metric": "cpu_usage", '
            '"threshold": 90, "severity": "high"}}'
        )
        assert _check_aa_enum_values(response) is True

    def test_aa_nested_objects_with_think(self):
        response = (
            "<think>The address needs to be a nested object.</think>\n"
            '{"name": "create_order", "arguments": '
            '{"product_id": "SKU-7890", "quantity": 3, '
            '"shipping_address": {"street": "456 Pine Ave", '
            '"city": "Seattle", "state": "WA", '
            '"zip": "98101", "country": "United States"}}}'
        )
        assert _check_aa_nested_objects(response) is True

    def test_aa_numeric_coercion_with_think(self):
        response = (
            "<think>Amount must be a number, not a string.</think>\n"
            '{"name": "transfer_funds", "arguments": '
            '{"from_account": "ACC-001", "to_account": "ACC-002", '
            '"amount": 1500, "currency": "USD"}}'
        )
        assert _check_aa_numeric_coercion(response) is True

    def test_aa_boolean_natural_lang_with_think(self):
        response = (
            "<think>Turn on = true, disable = false.</think>\n"
            '{"name": "update_notification_settings", "arguments": '
            '{"user_id": "usr_123", "email_notifications": true, '
            '"sms_notifications": false}}'
        )
        assert _check_aa_boolean_natural_lang(response) is True

    def test_aa_all_required_args_with_think(self):
        response = (
            "<think>Required: service_name, version, environment, replicas.</think>\n"
            '{"name": "deploy_service", "arguments": '
            '{"service_name": "auth-service", "version": "2.1.0", '
            '"environment": "staging", "replicas": 3}}'
        )
        assert _check_aa_all_required_args(response) is True

    def test_aa_preserve_exact_strings_with_think(self):
        response = (
            "<think>I must pass the command exactly as given.</think>\n"
            '{"name": "execute_command", "arguments": '
            '{"command": "grep -rn \'TODO\' src/"}}'
        )
        assert _check_aa_preserve_exact_strings(response) is True


# =============================================================================
# Hermes format tests (check functions should work with Hermes-wrapped calls)
# =============================================================================


class TestToolCallingHermesFormat:
    """Ensure check functions work with Hermes-format tool calls."""

    def test_selection_hermes_format(self):
        response = (
            "<tool_call>\n"
            '{"name": "search_flights", "arguments": '
            '{"origin": "New York", "destination": "London", "date": "2024-03-15"}}\n'
            "</tool_call>"
        )
        assert _check_ts_correct_tool(response) is True

    def test_accuracy_hermes_format(self):
        response = (
            "<tool_call>\n"
            '{"name": "schedule_event", "arguments": '
            '{"title": "Standup", "date": "2024-06-15", "time": "09:30"}}\n'
            "</tool_call>"
        )
        assert _check_aa_date_iso_format(response) is True


# =============================================================================
# JSON code block format tests
# =============================================================================


class TestToolCallingCodeBlockFormat:
    """Ensure check functions work with JSON code blocks."""

    def test_selection_code_block(self):
        response = (
            "I'll search for flights:\n\n"
            "```json\n"
            '{"name": "search_flights", "arguments": '
            '{"origin": "New York", "destination": "London", "date": "2024-03-15"}}\n'
            "```"
        )
        assert _check_ts_correct_tool(response) is True

    def test_accuracy_code_block(self):
        response = (
            "Here's the tool call:\n\n"
            "```json\n"
            '{"name": "create_order", "arguments": '
            '{"product_id": "SKU-7890", "quantity": 3, '
            '"shipping_address": {"street": "456 Pine Ave", '
            '"city": "Seattle", "state": "WA", '
            '"zip": "98101", "country": "US"}}}\n'
            "```"
        )
        assert _check_aa_nested_objects(response) is True
