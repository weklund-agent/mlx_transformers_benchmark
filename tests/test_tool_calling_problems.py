"""Tests for tool calling problems: all 5 subcategories.

Each subcategory has pass tests (correct responses) and fail tests (incorrect
responses), plus think-block-wrapped response tests.

Subcategories:
- Tool Selection (8): ts_* prefix
- Argument Accuracy (8): aa_* prefix
- Multi-Tool (8): mt_* prefix
- Edge Cases (8): ec_* prefix
- Format Compliance (8): fc_* prefix
"""

import json

import pytest

from mtb.quality_benchmarks.tool_calling_problems import (
    ARGUMENT_ACCURACY_PROBLEMS,
    EDGE_CASES_PROBLEMS,
    FORMAT_COMPLIANCE_PROBLEMS,
    MULTI_TOOL_PROBLEMS,
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
    _check_ec_deprecated_param,
    _check_ec_handle_tool_error,
    _check_ec_idempotency,
    _check_ec_missing_required_params,
    _check_ec_no_matching_tool,
    _check_ec_optional_params,
    _check_ec_refuse_trivial,
    _check_ec_reject_harmful,
    _check_fc_array_params,
    _check_fc_consistent_format,
    _check_fc_empty_string,
    _check_fc_multiple_tools_single_response,
    _check_fc_null_argument,
    _check_fc_optional_included,
    _check_fc_type_matching,
    _check_fc_valid_json_format,
    _check_mt_chain_of_three,
    _check_mt_conditional_planning,
    _check_mt_mixed,
    _check_mt_multi_turn_with_result,
    _check_mt_parallel_different,
    _check_mt_parallel_independent,
    _check_mt_sequential_dependent,
    _check_mt_three_parallel,
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
    """Structural validation for all 40 tool calling problems."""

    def test_total_problem_count(self):
        assert len(TOOL_CALLING_NEW_PROBLEMS) == 40

    def test_selection_count(self):
        assert len(TOOL_SELECTION_PROBLEMS) == 8

    def test_accuracy_count(self):
        assert len(ARGUMENT_ACCURACY_PROBLEMS) == 8

    def test_multi_tool_count(self):
        assert len(MULTI_TOOL_PROBLEMS) == 8

    def test_edge_cases_count(self):
        assert len(EDGE_CASES_PROBLEMS) == 8

    def test_format_compliance_count(self):
        assert len(FORMAT_COMPLIANCE_PROBLEMS) == 8

    def test_subcategory_distribution(self):
        """8 per subcategory based on name prefix."""
        from collections import Counter

        prefix_map = {
            "ts_": "tool_selection",
            "aa_": "argument_accuracy",
            "mt_": "multi_tool",
            "ec_": "edge_cases",
            "fc_": "format_compliance",
        }
        counts = Counter()
        for p in TOOL_CALLING_NEW_PROBLEMS:
            for prefix, subcat in prefix_map.items():
                if p.name.startswith(prefix):
                    counts[subcat] += 1
                    break
        assert counts == {
            "tool_selection": 8,
            "argument_accuracy": 8,
            "multi_tool": 8,
            "edge_cases": 8,
            "format_compliance": 8,
        }

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

    def test_multi_tool_names_have_mt_prefix(self):
        for p in MULTI_TOOL_PROBLEMS:
            assert p.name.startswith("mt_"), f"{p.name} should start with 'mt_'"

    def test_edge_cases_names_have_ec_prefix(self):
        for p in EDGE_CASES_PROBLEMS:
            assert p.name.startswith("ec_"), f"{p.name} should start with 'ec_'"

    def test_format_compliance_names_have_fc_prefix(self):
        for p in FORMAT_COMPLIANCE_PROBLEMS:
            assert p.name.startswith("fc_"), f"{p.name} should start with 'fc_'"

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

    def test_tool_calling_problems_is_40(self):
        """TOOL_CALLING_PROBLEMS in eval_problems.py has exactly 40 problems."""
        from mtb.quality_benchmarks.eval_problems import TOOL_CALLING_PROBLEMS

        assert len(TOOL_CALLING_PROBLEMS) == 40


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


# =============================================================================
# Multi-Tool – pass cases (VAL-TOOLCALL-300)
# =============================================================================


class TestMultiToolPass:
    """Correct responses for all 8 multi-tool problems."""

    def test_mt_parallel_independent_pass(self):
        response = json.dumps(
            [
                {"name": "get_weather", "arguments": {"city": "New York"}},
                {"name": "get_weather", "arguments": {"city": "San Francisco"}},
            ]
        )
        assert _check_mt_parallel_independent(response) is True

    def test_mt_sequential_dependent_pass(self):
        response = json.dumps(
            [
                {"name": "search_user", "arguments": {"email": "jane@example.com"}},
                {
                    "name": "get_order_history",
                    "arguments": {"user_id": "usr_123", "limit": 5},
                },
            ]
        )
        assert _check_mt_sequential_dependent(response) is True

    def test_mt_mixed_pass(self):
        response = json.dumps(
            [
                {"name": "get_weather", "arguments": {"city": "Tokyo"}},
                {
                    "name": "search_restaurants",
                    "arguments": {"city": "Tokyo", "cuisine": "sushi"},
                },
                {
                    "name": "book_restaurant",
                    "arguments": {
                        "restaurant_id": "r_001",
                        "party_size": 4,
                        "time": "19:00",
                    },
                },
            ]
        )
        assert _check_mt_mixed(response) is True

    def test_mt_three_parallel_pass(self):
        response = json.dumps(
            [
                {"name": "get_stock_price", "arguments": {"symbol": "AAPL"}},
                {"name": "get_stock_price", "arguments": {"symbol": "GOOGL"}},
                {"name": "get_stock_price", "arguments": {"symbol": "MSFT"}},
            ]
        )
        assert _check_mt_three_parallel(response) is True

    def test_mt_multi_turn_with_result_pass(self):
        response = json.dumps(
            {
                "name": "calculate",
                "arguments": {"expression": "85.50 * 0.20 + 85.50"},
            }
        )
        assert _check_mt_multi_turn_with_result(response) is True

    def test_mt_chain_of_three_pass(self):
        response = json.dumps(
            [
                {"name": "get_file_content", "arguments": {"path": "src/auth.py"}},
                {
                    "name": "analyze_code",
                    "arguments": {"code": "...", "language": "python"},
                },
                {
                    "name": "create_ticket",
                    "arguments": {
                        "title": "Bug in auth.py",
                        "description": "...",
                        "priority": "high",
                    },
                },
            ]
        )
        assert _check_mt_chain_of_three(response) is True

    def test_mt_parallel_different_pass(self):
        response = json.dumps(
            [
                {"name": "get_user_profile", "arguments": {"user_id": "USR-100"}},
                {"name": "get_account_balance", "arguments": {"account_id": "ACC-200"}},
            ]
        )
        assert _check_mt_parallel_different(response) is True

    def test_mt_conditional_planning_pass(self):
        response = (
            '{"name": "check_inventory", "arguments": {"product_id": "PROD-567"}}\n\n'
            "If the product is in stock, I will call:\n"
            '{"name": "place_order", "arguments": {"product_id": "PROD-567", '
            '"quantity": 2, "customer_id": "CUST-123"}}\n\n'
            "If the product is not available, I will call:\n"
            '{"name": "notify_restock", "arguments": {"product_id": "PROD-567", '
            '"customer_email": "buyer@example.com"}}'
        )
        assert _check_mt_conditional_planning(response) is True


# =============================================================================
# Multi-Tool – fail cases (VAL-TOOLCALL-308)
# =============================================================================


class TestMultiToolFail:
    """Incomplete or incorrect responses for multi-tool problems."""

    def test_mt_parallel_independent_only_one(self):
        response = json.dumps(
            {"name": "get_weather", "arguments": {"city": "New York"}}
        )
        assert _check_mt_parallel_independent(response) is False

    def test_mt_parallel_independent_wrong_cities(self):
        response = json.dumps(
            [
                {"name": "get_weather", "arguments": {"city": "Chicago"}},
                {"name": "get_weather", "arguments": {"city": "Boston"}},
            ]
        )
        assert _check_mt_parallel_independent(response) is False

    def test_mt_sequential_dependent_missing_search(self):
        response = json.dumps(
            {
                "name": "get_order_history",
                "arguments": {"user_id": "usr_123", "limit": 5},
            }
        )
        assert _check_mt_sequential_dependent(response) is False

    def test_mt_mixed_missing_book(self):
        response = json.dumps(
            [
                {"name": "get_weather", "arguments": {"city": "Tokyo"}},
                {
                    "name": "search_restaurants",
                    "arguments": {"city": "Tokyo", "cuisine": "sushi"},
                },
            ]
        )
        assert _check_mt_mixed(response) is False

    def test_mt_three_parallel_only_two(self):
        response = json.dumps(
            [
                {"name": "get_stock_price", "arguments": {"symbol": "AAPL"}},
                {"name": "get_stock_price", "arguments": {"symbol": "GOOGL"}},
            ]
        )
        assert _check_mt_three_parallel(response) is False

    def test_mt_multi_turn_wrong_expression(self):
        response = json.dumps(
            {"name": "calculate", "arguments": {"expression": "2 + 2"}}
        )
        assert _check_mt_multi_turn_with_result(response) is False

    def test_mt_chain_of_three_missing_analyze(self):
        response = json.dumps(
            [
                {"name": "get_file_content", "arguments": {"path": "src/auth.py"}},
                {
                    "name": "create_ticket",
                    "arguments": {
                        "title": "Bug",
                        "description": "...",
                        "priority": "high",
                    },
                },
            ]
        )
        assert _check_mt_chain_of_three(response) is False

    def test_mt_parallel_different_only_profile(self):
        response = json.dumps(
            {"name": "get_user_profile", "arguments": {"user_id": "USR-100"}}
        )
        assert _check_mt_parallel_different(response) is False

    def test_mt_conditional_planning_no_check(self):
        response = (
            "I'll place the order directly.\n"
            '{"name": "place_order", "arguments": {"product_id": "PROD-567", '
            '"quantity": 2, "customer_id": "CUST-123"}}'
        )
        assert _check_mt_conditional_planning(response) is False


# =============================================================================
# Edge Cases – pass cases (VAL-TOOLCALL-400)
# =============================================================================


class TestEdgeCasesPass:
    """Correct responses for all 8 edge case problems."""

    def test_ec_refuse_trivial_pass(self):
        response = "The answer is 4."
        assert _check_ec_refuse_trivial(response) is True

    def test_ec_refuse_trivial_pass_verbose(self):
        response = "2 + 2 = 4. That's a simple arithmetic fact."
        assert _check_ec_refuse_trivial(response) is True

    def test_ec_missing_required_params_pass(self):
        response = (
            "I'd be happy to help book a flight from Chicago on December 20th, "
            "but I need to know the destination. Where would you like to fly to?"
        )
        assert _check_ec_missing_required_params(response) is True

    def test_ec_handle_tool_error_pass(self):
        response = (
            "I'm sorry, the weather service returned an error (HTTP 503). "
            "Let me try again in a moment, or you could check weather.com as an alternative."
        )
        assert _check_ec_handle_tool_error(response) is True

    def test_ec_no_matching_tool_pass(self):
        response = (
            "I'm sorry, but I don't have a tool available to book restaurants. "
            "The tools I have access to are for weather, currency conversion, "
            "and translation. You could try OpenTable or call the restaurant directly."
        )
        assert _check_ec_no_matching_tool(response) is True

    def test_ec_optional_params_pass(self):
        response = json.dumps(
            {"name": "search_products", "arguments": {"query": "laptops"}}
        )
        assert _check_ec_optional_params(response) is True

    def test_ec_optional_params_pass_with_extras(self):
        """Including optional params is also fine, as long as query has 'laptop'."""
        response = json.dumps(
            {
                "name": "search_products",
                "arguments": {"query": "laptop", "sort_by": "rating"},
            }
        )
        assert _check_ec_optional_params(response) is True

    def test_ec_reject_harmful_pass(self):
        response = (
            "I cannot execute that command. 'rm -rf /' is a dangerous command that "
            "would delete all files on the server. I refuse to execute destructive commands."
        )
        assert _check_ec_reject_harmful(response) is True

    def test_ec_deprecated_param_pass(self):
        response = json.dumps(
            {
                "name": "send_notification",
                "arguments": {
                    "recipient_email": "admin@company.com",
                    "message": "Server maintenance scheduled for tonight.",
                },
            }
        )
        assert _check_ec_deprecated_param(response) is True

    def test_ec_idempotency_pass(self):
        response = (
            "It looks like the user account for bob (bob@test.com) was already "
            "created in the previous step. Creating it again would result in a "
            "duplicate. Would you like me to proceed anyway?"
        )
        assert _check_ec_idempotency(response) is True


# =============================================================================
# Edge Cases – fail cases (VAL-TOOLCALL-408)
# =============================================================================


class TestEdgeCasesFail:
    """Responses that violate edge-case expectations."""

    def test_ec_refuse_trivial_calls_tool(self):
        """Should not call a tool for 2+2."""
        response = json.dumps(
            {"name": "calculate", "arguments": {"expression": "2 + 2"}}
        )
        assert _check_ec_refuse_trivial(response) is False

    def test_ec_missing_required_params_guesses(self):
        """Should not guess the destination."""
        response = json.dumps(
            {
                "name": "book_flight",
                "arguments": {
                    "origin": "Chicago",
                    "destination": "New York",
                    "date": "2024-12-20",
                },
            }
        )
        assert _check_ec_missing_required_params(response) is False

    def test_ec_handle_tool_error_ignores(self):
        """Just repeating the request without acknowledging the error."""
        response = "The weather in Paris is sunny and 22°C."
        assert _check_ec_handle_tool_error(response) is False

    def test_ec_no_matching_tool_calls_unrelated(self):
        """Should not call an unrelated tool."""
        response = json.dumps({"name": "get_weather", "arguments": {"city": "Seattle"}})
        assert _check_ec_no_matching_tool(response) is False

    def test_ec_optional_params_wrong_query(self):
        """Query doesn't mention laptops."""
        response = json.dumps(
            {"name": "search_products", "arguments": {"query": "phones"}}
        )
        assert _check_ec_optional_params(response) is False

    def test_ec_reject_harmful_executes_command(self):
        """Should not execute the dangerous command."""
        response = json.dumps(
            {"name": "execute_command", "arguments": {"command": "rm -rf /"}}
        )
        assert _check_ec_reject_harmful(response) is False

    def test_ec_deprecated_param_uses_old_field(self):
        """Uses deprecated 'email' field instead of 'recipient_email'."""
        response = json.dumps(
            {
                "name": "send_notification",
                "arguments": {
                    "email": "admin@company.com",
                    "message": "Server maintenance scheduled for tonight.",
                },
            }
        )
        assert _check_ec_deprecated_param(response) is False

    def test_ec_idempotency_creates_again(self):
        """Blindly creates the user again without acknowledging the duplicate."""
        response = json.dumps(
            {
                "name": "create_user",
                "arguments": {
                    "username": "bob",
                    "email": "bob@test.com",
                    "role": "admin",
                },
            }
        )
        assert _check_ec_idempotency(response) is False


# =============================================================================
# Format Compliance – pass cases (VAL-TOOLCALL-500)
# =============================================================================


class TestFormatCompliancePass:
    """Correct format for all 8 format compliance problems."""

    def test_fc_valid_json_format_pass(self):
        response = json.dumps({"name": "get_time", "arguments": {"timezone": "UTC"}})
        assert _check_fc_valid_json_format(response) is True

    def test_fc_array_params_pass(self):
        response = json.dumps(
            {
                "name": "create_post",
                "arguments": {
                    "title": "Python Tips",
                    "content": "Here are some useful Python tips.",
                    "tags": ["python", "tutorial"],
                },
            }
        )
        assert _check_fc_array_params(response) is True

    def test_fc_optional_included_pass(self):
        response = json.dumps(
            {
                "name": "send_email",
                "arguments": {
                    "to": "john@company.com",
                    "subject": "Meeting Update",
                    "body": "The meeting has been moved to 3pm.",
                    "cc": ["manager@company.com"],
                },
            }
        )
        assert _check_fc_optional_included(response) is True

    def test_fc_multiple_tools_single_response_pass(self):
        response = json.dumps(
            [
                {
                    "name": "set_alarm",
                    "arguments": {"time": "07:00", "label": "Wake up"},
                },
                {"name": "get_weather", "arguments": {"city": "Boston"}},
            ]
        )
        assert _check_fc_multiple_tools_single_response(response) is True

    def test_fc_null_argument_pass(self):
        response = json.dumps(
            {
                "name": "update_profile",
                "arguments": {"display_name": "Alex", "bio": None},
            }
        )
        assert _check_fc_null_argument(response) is True

    def test_fc_null_argument_pass_without_bio(self):
        """Not including bio at all is also acceptable."""
        response = json.dumps(
            {"name": "update_profile", "arguments": {"display_name": "Alex"}}
        )
        assert _check_fc_null_argument(response) is True

    def test_fc_empty_string_pass(self):
        response = json.dumps(
            {
                "name": "list_files",
                "arguments": {"directory": "/home/user/documents", "filter": ""},
            }
        )
        assert _check_fc_empty_string(response) is True

    def test_fc_empty_string_pass_without_filter(self):
        """Not including filter at all is also acceptable."""
        response = json.dumps(
            {"name": "list_files", "arguments": {"directory": "/home/user/documents"}}
        )
        assert _check_fc_empty_string(response) is True

    def test_fc_consistent_format_pass(self):
        response = json.dumps(
            [
                {"name": "create_folder", "arguments": {"path": "/archive"}},
                {
                    "name": "move_file",
                    "arguments": {
                        "source": "/data/old_report.csv",
                        "destination": "/archive/old_report.csv",
                    },
                },
            ]
        )
        assert _check_fc_consistent_format(response) is True

    def test_fc_type_matching_pass(self):
        response = json.dumps(
            {
                "name": "configure_server",
                "arguments": {
                    "hostname": "api.example.com",
                    "port": 8080,
                    "ssl_enabled": True,
                    "allowed_ips": ["10.0.0.1", "10.0.0.2"],
                },
            }
        )
        assert _check_fc_type_matching(response) is True


# =============================================================================
# Format Compliance – fail cases (VAL-TOOLCALL-508)
# =============================================================================


class TestFormatComplianceFail:
    """Incorrect format for format compliance problems."""

    def test_fc_valid_json_format_prose(self):
        """Prose description instead of structured JSON."""
        response = "I would call get_time with timezone set to UTC."
        assert _check_fc_valid_json_format(response) is False

    def test_fc_valid_json_format_wrong_timezone(self):
        response = json.dumps(
            {"name": "get_time", "arguments": {"timezone": "US/Eastern"}}
        )
        assert _check_fc_valid_json_format(response) is False

    def test_fc_array_params_string_tags(self):
        """Tags as a comma-separated string instead of array."""
        response = json.dumps(
            {
                "name": "create_post",
                "arguments": {
                    "title": "Python Tips",
                    "content": "...",
                    "tags": "python, tutorial",
                },
            }
        )
        assert _check_fc_array_params(response) is False

    def test_fc_array_params_missing_tag(self):
        """Only one of the two required tags."""
        response = json.dumps(
            {
                "name": "create_post",
                "arguments": {
                    "title": "Python Tips",
                    "content": "...",
                    "tags": ["python"],
                },
            }
        )
        assert _check_fc_array_params(response) is False

    def test_fc_optional_included_no_cc(self):
        """Missing the CC that was explicitly requested."""
        response = json.dumps(
            {
                "name": "send_email",
                "arguments": {
                    "to": "john@company.com",
                    "subject": "Meeting Update",
                    "body": "The meeting has been moved to 3pm.",
                },
            }
        )
        assert _check_fc_optional_included(response) is False

    def test_fc_multiple_tools_only_one(self):
        """Only one of two required tools."""
        response = json.dumps(
            {"name": "set_alarm", "arguments": {"time": "07:00", "label": "Wake up"}}
        )
        assert _check_fc_multiple_tools_single_response(response) is False

    def test_fc_null_argument_wrong_name(self):
        """Wrong display name."""
        response = json.dumps(
            {
                "name": "update_profile",
                "arguments": {"display_name": "Bob", "bio": None},
            }
        )
        assert _check_fc_null_argument(response) is False

    def test_fc_empty_string_wrong_directory(self):
        """Wrong directory."""
        response = json.dumps(
            {"name": "list_files", "arguments": {"directory": "/tmp"}}
        )
        assert _check_fc_empty_string(response) is False

    def test_fc_consistent_format_missing_tool(self):
        """Only one of two required tools."""
        response = json.dumps(
            {"name": "create_folder", "arguments": {"path": "/archive"}}
        )
        assert _check_fc_consistent_format(response) is False

    def test_fc_type_matching_string_port(self):
        """Port as string instead of integer."""
        response = (
            '{"name": "configure_server", "arguments": '
            '{"hostname": "api.example.com", "port": "8080", '
            '"ssl_enabled": true, "allowed_ips": ["10.0.0.1"]}}'
        )
        assert _check_fc_type_matching(response) is False

    def test_fc_type_matching_string_boolean(self):
        """SSL as string instead of boolean."""
        response = (
            '{"name": "configure_server", "arguments": '
            '{"hostname": "api.example.com", "port": 8080, '
            '"ssl_enabled": "true", "allowed_ips": ["10.0.0.1"]}}'
        )
        assert _check_fc_type_matching(response) is False

    def test_fc_type_matching_string_array(self):
        """IPs as comma-separated string instead of array."""
        response = (
            '{"name": "configure_server", "arguments": '
            '{"hostname": "api.example.com", "port": 8080, '
            '"ssl_enabled": true, "allowed_ips": "10.0.0.1, 10.0.0.2"}}'
        )
        assert _check_fc_type_matching(response) is False


# =============================================================================
# Multi-Tool – think block tests
# =============================================================================


class TestMultiToolWithThinkBlocks:
    """Think-block-wrapped responses should still pass checks."""

    def test_mt_parallel_independent_with_think(self):
        response = (
            "<think>User wants weather for two cities. Two parallel calls needed.</think>\n"
            '[{"name": "get_weather", "arguments": {"city": "New York"}}, '
            '{"name": "get_weather", "arguments": {"city": "San Francisco"}}]'
        )
        assert _check_mt_parallel_independent(response) is True

    def test_mt_sequential_dependent_with_think(self):
        response = (
            "<think>Need to find user first, then get orders.</think>\n"
            '[{"name": "search_user", "arguments": {"email": "jane@example.com"}}, '
            '{"name": "get_order_history", "arguments": {"user_id": "...", "limit": 5}}]'
        )
        assert _check_mt_sequential_dependent(response) is True

    def test_mt_three_parallel_with_think(self):
        response = (
            "<think>Three stock lookups, all independent.</think>\n"
            '[{"name": "get_stock_price", "arguments": {"symbol": "AAPL"}}, '
            '{"name": "get_stock_price", "arguments": {"symbol": "GOOGL"}}, '
            '{"name": "get_stock_price", "arguments": {"symbol": "MSFT"}}]'
        )
        assert _check_mt_three_parallel(response) is True

    def test_mt_multi_turn_with_think(self):
        response = (
            "<think>85.50 * 0.20 = 17.10, total = 102.60.</think>\n"
            '{"name": "calculate", "arguments": {"expression": "85.50 * 1.20"}}'
        )
        assert _check_mt_multi_turn_with_result(response) is True


# =============================================================================
# Edge Cases – think block tests
# =============================================================================


class TestEdgeCasesWithThinkBlocks:
    """Think-block-wrapped responses should still pass checks."""

    def test_ec_refuse_trivial_with_think(self):
        response = (
            "<think>2+2 is trivial, no need for a tool.</think>\n" "The answer is 4."
        )
        assert _check_ec_refuse_trivial(response) is True

    def test_ec_missing_required_with_think(self):
        response = (
            "<think>The user didn't specify a destination.</think>\n"
            "I can help book a flight from Chicago on December 20th. "
            "Where would you like to fly to (destination)?"
        )
        assert _check_ec_missing_required_params(response) is True

    def test_ec_no_matching_tool_with_think(self):
        response = (
            "<think>None of the tools can book restaurants.</think>\n"
            "I'm sorry, I don't have a tool available to book restaurants."
        )
        assert _check_ec_no_matching_tool(response) is True

    def test_ec_reject_harmful_with_think(self):
        response = (
            "<think>This is a dangerous command. Must refuse.</think>\n"
            "I cannot execute 'rm -rf /' as it is an extremely dangerous "
            "and destructive command that would delete everything."
        )
        assert _check_ec_reject_harmful(response) is True

    def test_ec_deprecated_param_with_think(self):
        response = (
            "<think>The email field is deprecated, use recipient_email.</think>\n"
            '{"name": "send_notification", "arguments": '
            '{"recipient_email": "admin@company.com", '
            '"message": "Server maintenance scheduled for tonight."}}'
        )
        assert _check_ec_deprecated_param(response) is True

    def test_ec_idempotency_with_think(self):
        response = (
            "<think>The same user was already created in the previous turn.</think>\n"
            "It looks like the user bob with email bob@test.com was already "
            "created. Creating a duplicate account is not recommended."
        )
        assert _check_ec_idempotency(response) is True


# =============================================================================
# Format Compliance – think block tests
# =============================================================================


class TestFormatComplianceWithThinkBlocks:
    """Think-block-wrapped responses should still pass checks."""

    def test_fc_valid_json_with_think(self):
        response = (
            "<think>User wants UTC time.</think>\n"
            '{"name": "get_time", "arguments": {"timezone": "UTC"}}'
        )
        assert _check_fc_valid_json_format(response) is True

    def test_fc_array_params_with_think(self):
        response = (
            "<think>Tags must be an array.</think>\n"
            '{"name": "create_post", "arguments": '
            '{"title": "Python Tips", "content": "...", '
            '"tags": ["python", "tutorial"]}}'
        )
        assert _check_fc_array_params(response) is True

    def test_fc_type_matching_with_think(self):
        response = (
            "<think>Port must be integer, ssl must be boolean, ips must be array.</think>\n"
            '{"name": "configure_server", "arguments": '
            '{"hostname": "api.example.com", "port": 8080, '
            '"ssl_enabled": true, "allowed_ips": ["10.0.0.1", "10.0.0.2"]}}'
        )
        assert _check_fc_type_matching(response) is True

    def test_fc_multiple_tools_with_think(self):
        response = (
            "<think>Need both alarm and weather calls.</think>\n"
            '[{"name": "set_alarm", "arguments": {"time": "07:00", "label": "Wake up"}}, '
            '{"name": "get_weather", "arguments": {"city": "Boston"}}]'
        )
        assert _check_fc_multiple_tools_single_response(response) is True
