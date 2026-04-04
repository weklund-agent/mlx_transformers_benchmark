"""Tool calling evaluation problems for measuring model quality.

Contains 16 problems across 2 subcategories:
- Tool Selection (8): Choose the correct tool from multiple options
- Argument Accuracy (8): Extract and format arguments correctly

Each check function uses the tool_call_parser internally for structured
validation rather than simple keyword matching.
"""

import json
from typing import List, Optional

from mtb.quality_benchmarks.eval_problem import EvalProblem
from mtb.quality_benchmarks.tool_call_parser import ToolCall, parse_tool_calls
from mtb.quality_benchmarks.utils import _strip_thinking


# =============================================================================
# Helper: parse and match tool calls
# =============================================================================


def _parse_response_tool_calls(response: str) -> Optional[list[ToolCall]]:
    """Strip thinking blocks and parse tool calls from a response."""
    cleaned = _strip_thinking(response)
    return parse_tool_calls(cleaned)


# =============================================================================
# Category 1: Tool Selection (8 problems)
# =============================================================================

# --- 1. Correct tool from 5-8 options ---


def _check_ts_correct_tool(response: str) -> bool:
    """Check model selects 'search_flights' from 6 travel-related tools."""
    calls = _parse_response_tool_calls(response)
    if not calls:
        return False
    return any(c.name == "search_flights" for c in calls)


TS_CORRECT_TOOL = EvalProblem(
    category="tool_calling",
    name="ts_correct_tool",
    prompt=(
        "You have access to the following tools:\n\n"
        "1. search_flights(origin: str, destination: str, date: str) "
        "- Search for available flights\n"
        "2. book_hotel(city: str, check_in: str, check_out: str) "
        "- Book a hotel room\n"
        "3. rent_car(city: str, pickup_date: str, return_date: str) "
        "- Rent a car\n"
        "4. get_weather_forecast(city: str, date: str) "
        "- Get weather forecast for a city\n"
        "5. convert_currency(amount: float, from_currency: str, to_currency: str) "
        "- Convert between currencies\n"
        "6. translate_text(text: str, target_language: str) "
        "- Translate text to another language\n\n"
        "User: I need to find flights from New York to London on March 15th.\n\n"
        "Call the appropriate tool with the correct arguments. "
        "Output your response as a JSON tool call."
    ),
    check=_check_ts_correct_tool,
    max_tokens=256,
)


# --- 2. Ambiguous request ---


def _check_ts_ambiguous_request(response: str) -> bool:
    """Check model selects 'schedule_meeting' for a scheduling request.

    The user says 'set up a call with the marketing team for next Tuesday'.
    'schedule_meeting' is the right tool, not 'send_message' or 'set_reminder'.
    """
    calls = _parse_response_tool_calls(response)
    if not calls:
        return False
    return any(c.name == "schedule_meeting" for c in calls)


TS_AMBIGUOUS_REQUEST = EvalProblem(
    category="tool_calling",
    name="ts_ambiguous_request",
    prompt=(
        "You have access to the following tools:\n\n"
        "1. send_message(recipient: str, message: str) "
        "- Send a text message to someone\n"
        "2. schedule_meeting(title: str, participants: list, date: str, time: str) "
        "- Schedule a meeting with participants\n"
        "3. set_reminder(message: str, datetime: str) "
        "- Set a personal reminder\n"
        "4. create_task(title: str, assignee: str, due_date: str) "
        "- Create a task for someone\n"
        "5. send_email(to: str, subject: str, body: str) "
        "- Send an email\n\n"
        "User: Set up a call with the marketing team for next Tuesday at 2pm.\n\n"
        "Call the appropriate tool with the correct arguments. "
        "Output your response as a JSON tool call."
    ),
    check=_check_ts_ambiguous_request,
    max_tokens=256,
)


# --- 3. 'None' selection (no tool appropriate) ---


def _check_ts_none_selection(response: str) -> bool:
    """Check that model does NOT call any tool for a greeting.

    The user says 'Hello, how are you?'. No tool should be called.
    A correct response answers directly without tool calls.
    """
    cleaned = _strip_thinking(response)
    calls = parse_tool_calls(cleaned)
    # Should NOT produce tool calls — the response should be direct
    if calls and len(calls) > 0:
        return False
    # Check for a direct greeting response (not empty)
    return len(cleaned.strip()) > 0


TS_NONE_SELECTION = EvalProblem(
    category="tool_calling",
    name="ts_none_selection",
    prompt=(
        "You have access to the following tools:\n\n"
        "1. search_database(query: str) - Search the database for records\n"
        "2. create_record(table: str, data: dict) - Create a new database record\n"
        "3. delete_record(table: str, id: str) - Delete a database record\n"
        "4. update_record(table: str, id: str, data: dict) - Update a record\n"
        "5. generate_report(report_type: str, date_range: str) "
        "- Generate a business report\n\n"
        "User: Hello, how are you today?\n\n"
        "If a tool is appropriate, call it. If not, respond directly without "
        "using any tools. Do NOT call a tool unless the user's request "
        "clearly requires one."
    ),
    check=_check_ts_none_selection,
    max_tokens=256,
)


# --- 4. Similar tool names ---


def _check_ts_similar_names(response: str) -> bool:
    """Check model picks 'get_user_profile' NOT 'get_user_preferences' or 'get_user_activity'."""
    calls = _parse_response_tool_calls(response)
    if not calls:
        return False
    return any(c.name == "get_user_profile" for c in calls)


TS_SIMILAR_NAMES = EvalProblem(
    category="tool_calling",
    name="ts_similar_names",
    prompt=(
        "You have access to the following tools:\n\n"
        "1. get_user_profile(user_id: str) "
        "- Retrieve a user's profile information (name, email, bio)\n"
        "2. get_user_preferences(user_id: str) "
        "- Retrieve a user's app preferences (theme, language, notifications)\n"
        "3. get_user_activity(user_id: str, days: int) "
        "- Retrieve a user's recent activity log\n"
        "4. get_user_permissions(user_id: str) "
        "- Retrieve a user's access permissions and roles\n"
        "5. get_user_billing(user_id: str) "
        "- Retrieve a user's billing and subscription details\n\n"
        "User: I need to see the profile details for user 'alice_42' — "
        "specifically their name and email.\n\n"
        "Call the appropriate tool. Output your response as a JSON tool call."
    ),
    check=_check_ts_similar_names,
    max_tokens=256,
)


# --- 5. Parameter-based selection ---


def _check_ts_parameter_based(response: str) -> bool:
    """Check model picks 'resize_image' which accepts width and height params.

    'compress_image' and 'crop_image' are available but don't match the request.
    """
    calls = _parse_response_tool_calls(response)
    if not calls:
        return False
    return any(c.name == "resize_image" for c in calls)


TS_PARAMETER_BASED = EvalProblem(
    category="tool_calling",
    name="ts_parameter_based",
    prompt=(
        "You have access to the following tools:\n\n"
        "1. compress_image(file_path: str, quality: int) "
        "- Reduce image file size by adjusting quality (1-100)\n"
        "2. resize_image(file_path: str, width: int, height: int) "
        "- Change image dimensions to specific width and height\n"
        "3. crop_image(file_path: str, x: int, y: int, w: int, h: int) "
        "- Crop a rectangular region from an image\n"
        "4. rotate_image(file_path: str, degrees: int) "
        "- Rotate an image by the specified degrees\n"
        "5. convert_image(file_path: str, output_format: str) "
        "- Convert image to a different format (png, jpg, webp)\n\n"
        "User: I need to change my logo.png to be exactly 200 pixels wide "
        "and 100 pixels tall.\n\n"
        "Call the appropriate tool. Output your response as a JSON tool call."
    ),
    check=_check_ts_parameter_based,
    max_tokens=256,
)


# --- 6. Nested/complex tool descriptions ---


def _check_ts_nested_descriptions(response: str) -> bool:
    """Check model picks 'execute_query' for a SQL query request."""
    calls = _parse_response_tool_calls(response)
    if not calls:
        return False
    return any(c.name == "execute_query" for c in calls)


TS_NESTED_DESCRIPTIONS = EvalProblem(
    category="tool_calling",
    name="ts_nested_descriptions",
    prompt=(
        "You have access to the following tools:\n\n"
        "```json\n"
        "[\n"
        '  {"type": "function", "function": {"name": "execute_query", '
        '"description": "Execute a SQL query against the database and return results", '
        '"parameters": {"type": "object", "properties": {'
        '"query": {"type": "string", "description": "The SQL query to execute"}, '
        '"database": {"type": "string", "enum": ["production", "staging", "analytics"], '
        '"description": "Target database"}, '
        '"timeout_sec": {"type": "integer", "description": "Query timeout in seconds", '
        '"default": 30}}, '
        '"required": ["query", "database"]}}},\n'
        '  {"type": "function", "function": {"name": "list_tables", '
        '"description": "List all tables in a database", '
        '"parameters": {"type": "object", "properties": {'
        '"database": {"type": "string", "enum": ["production", "staging", "analytics"]}}, '
        '"required": ["database"]}}},\n'
        '  {"type": "function", "function": {"name": "describe_table", '
        '"description": "Get the schema of a specific table", '
        '"parameters": {"type": "object", "properties": {'
        '"database": {"type": "string"}, "table_name": {"type": "string"}}, '
        '"required": ["database", "table_name"]}}},\n'
        '  {"type": "function", "function": {"name": "export_results", '
        '"description": "Export query results to CSV file", '
        '"parameters": {"type": "object", "properties": {'
        '"query_id": {"type": "string"}, "format": {"type": "string", '
        '"enum": ["csv", "json", "parquet"]}}, '
        '"required": ["query_id", "format"]}}},\n'
        '  {"type": "function", "function": {"name": "create_view", '
        '"description": "Create a SQL view from a query", '
        '"parameters": {"type": "object", "properties": {'
        '"view_name": {"type": "string"}, "query": {"type": "string"}, '
        '"database": {"type": "string"}}, '
        '"required": ["view_name", "query", "database"]}}}\n'
        "]\n"
        "```\n\n"
        "User: Run this query on the analytics database: "
        "SELECT COUNT(*) FROM users WHERE created_at > '2024-01-01'\n\n"
        "Call the appropriate tool. Output your response as a JSON tool call."
    ),
    check=_check_ts_nested_descriptions,
    max_tokens=256,
)


# --- 7. Specialized vs general tool ---


def _check_ts_specialized_vs_general(response: str) -> bool:
    """Check model picks 'translate_document' (specialized) over 'translate_text' (general)."""
    calls = _parse_response_tool_calls(response)
    if not calls:
        return False
    return any(c.name == "translate_document" for c in calls)


TS_SPECIALIZED_VS_GENERAL = EvalProblem(
    category="tool_calling",
    name="ts_specialized_vs_general",
    prompt=(
        "You have access to the following tools:\n\n"
        "1. translate_text(text: str, source_lang: str, target_lang: str) "
        "- Translate a short text snippet between languages. "
        "Best for single sentences or phrases.\n"
        "2. translate_document(file_path: str, source_lang: str, target_lang: str, "
        "preserve_formatting: bool) "
        "- Translate an entire document while preserving its formatting, "
        "headers, and structure. Designed for long documents.\n"
        "3. detect_language(text: str) "
        "- Detect the language of a given text\n"
        "4. spell_check(text: str, language: str) "
        "- Check text for spelling errors\n"
        "5. summarize_text(text: str, max_length: int) "
        "- Summarize a long text into a shorter version\n\n"
        "User: I have a 50-page PDF report in German that needs to be translated "
        "to English. The formatting and headers must be preserved.\n\n"
        "Call the appropriate tool. Output your response as a JSON tool call."
    ),
    check=_check_ts_specialized_vs_general,
    max_tokens=256,
)


# --- 8. Multiple valid tools (pick best) ---


def _check_ts_multiple_valid(response: str) -> bool:
    """Check model picks 'create_ticket' for a bug report, not 'send_email' or 'send_message'.

    While send_email or send_message could relay the info, create_ticket
    is the most appropriate tool for logging a bug.
    """
    calls = _parse_response_tool_calls(response)
    if not calls:
        return False
    return any(c.name == "create_ticket" for c in calls)


TS_MULTIPLE_VALID = EvalProblem(
    category="tool_calling",
    name="ts_multiple_valid",
    prompt=(
        "You have access to the following tools:\n\n"
        "1. create_ticket(title: str, description: str, priority: str, "
        "assignee: str) "
        "- Create a bug/issue ticket in the project tracker\n"
        "2. send_email(to: str, subject: str, body: str) "
        "- Send an email to someone\n"
        "3. send_message(channel: str, message: str) "
        "- Send a message to a Slack channel\n"
        "4. create_document(title: str, content: str) "
        "- Create a new document\n"
        "5. add_comment(ticket_id: str, comment: str) "
        "- Add a comment to an existing ticket\n"
        "6. assign_task(task_id: str, assignee: str) "
        "- Assign a task to someone\n\n"
        "User: There's a critical bug in the checkout flow — customers "
        "are getting a 500 error when they click 'Pay Now'. "
        "Please log this as a high-priority issue and assign it to the "
        "backend team.\n\n"
        "Call the most appropriate tool. Output your response as a JSON tool call."
    ),
    check=_check_ts_multiple_valid,
    max_tokens=256,
)


# =============================================================================
# Category 2: Argument Accuracy (8 problems)
# =============================================================================

# --- 1. Date in ISO format ---


def _check_aa_date_iso_format(response: str) -> bool:
    """Check model provides date argument in ISO format (2024-06-15)."""
    calls = _parse_response_tool_calls(response)
    if not calls:
        return False
    for call in calls:
        if call.name == "schedule_event":
            date_val = call.arguments.get("date", "")
            # Accept ISO format: YYYY-MM-DD
            if isinstance(date_val, str) and date_val == "2024-06-15":
                return True
    return False


AA_DATE_ISO_FORMAT = EvalProblem(
    category="tool_calling",
    name="aa_date_iso_format",
    prompt=(
        "You have access to the following tool:\n\n"
        "```json\n"
        '{"type": "function", "function": {"name": "schedule_event", '
        '"description": "Schedule a calendar event", '
        '"parameters": {"type": "object", "properties": {'
        '"title": {"type": "string", "description": "Event title"}, '
        '"date": {"type": "string", "format": "date", '
        '"description": "Event date in ISO 8601 format (YYYY-MM-DD)"}, '
        '"time": {"type": "string", "description": "Event time in HH:MM format"}, '
        '"duration_minutes": {"type": "integer", "description": "Duration in minutes"}}, '
        '"required": ["title", "date", "time"]}}}\n'
        "```\n\n"
        "User: Schedule a team standup for June 15th, 2024 at 9:30 AM "
        "for 30 minutes.\n\n"
        "Call the tool with properly formatted arguments. "
        "Output your response as a JSON tool call."
    ),
    check=_check_aa_date_iso_format,
    max_tokens=256,
)


# --- 2. Email extraction ---


def _check_aa_email_extraction(response: str) -> bool:
    """Check model extracts email addresses correctly from natural language."""
    calls = _parse_response_tool_calls(response)
    if not calls:
        return False
    for call in calls:
        if call.name == "send_invitation":
            recipients = call.arguments.get("recipients") or call.arguments.get(
                "emails"
            )
            if isinstance(recipients, list):
                expected = {"alice@company.com", "bob@company.com"}
                actual = {r.strip().lower() for r in recipients if isinstance(r, str)}
                if expected.issubset(actual):
                    return True
            # Also accept a single string with both emails
            elif isinstance(recipients, str):
                lower = recipients.lower()
                if "alice@company.com" in lower and "bob@company.com" in lower:
                    return True
    return False


AA_EMAIL_EXTRACTION = EvalProblem(
    category="tool_calling",
    name="aa_email_extraction",
    prompt=(
        "You have access to the following tool:\n\n"
        "```json\n"
        '{"type": "function", "function": {"name": "send_invitation", '
        '"description": "Send meeting invitations to multiple recipients", '
        '"parameters": {"type": "object", "properties": {'
        '"event_title": {"type": "string", "description": "Title of the event"}, '
        '"recipients": {"type": "array", "items": {"type": "string", "format": "email"}, '
        '"description": "List of recipient email addresses"}, '
        '"message": {"type": "string", "description": "Optional invitation message"}}, '
        '"required": ["event_title", "recipients"]}}}\n'
        "```\n\n"
        "User: Send a meeting invitation for 'Q2 Planning' to Alice "
        "(alice@company.com) and Bob (bob@company.com). "
        "Include a note saying 'Please review the agenda beforehand.'\n\n"
        "Call the tool with properly formatted arguments. "
        "Output your response as a JSON tool call."
    ),
    check=_check_aa_email_extraction,
    max_tokens=256,
)


# --- 3. Enum values ---


def _check_aa_enum_values(response: str) -> bool:
    """Check model uses correct enum value 'high' for priority."""
    calls = _parse_response_tool_calls(response)
    if not calls:
        return False
    for call in calls:
        if call.name == "create_alert":
            severity = call.arguments.get("severity", "")
            if isinstance(severity, str) and severity.lower() == "high":
                return True
    return False


AA_ENUM_VALUES = EvalProblem(
    category="tool_calling",
    name="aa_enum_values",
    prompt=(
        "You have access to the following tool:\n\n"
        "```json\n"
        '{"type": "function", "function": {"name": "create_alert", '
        '"description": "Create a monitoring alert for a service", '
        '"parameters": {"type": "object", "properties": {'
        '"service_name": {"type": "string", "description": "Name of the service to monitor"}, '
        '"metric": {"type": "string", "description": "Metric to monitor (e.g., cpu_usage, memory, latency)"}, '
        '"threshold": {"type": "number", "description": "Threshold value that triggers the alert"}, '
        '"severity": {"type": "string", "enum": ["low", "medium", "high", "critical"], '
        '"description": "Alert severity level"}}, '
        '"required": ["service_name", "metric", "threshold", "severity"]}}}\n'
        "```\n\n"
        "User: Create a high-severity alert for the payment-service when "
        "CPU usage exceeds 90 percent.\n\n"
        "Call the tool with properly formatted arguments. "
        "Output your response as a JSON tool call."
    ),
    check=_check_aa_enum_values,
    max_tokens=256,
)


# --- 4. Nested objects ---


def _check_aa_nested_objects(response: str) -> bool:
    """Check model constructs nested address object correctly."""
    calls = _parse_response_tool_calls(response)
    if not calls:
        return False
    for call in calls:
        if call.name == "create_order":
            shipping = call.arguments.get("shipping_address")
            if not isinstance(shipping, dict):
                continue
            city = shipping.get("city", "")
            state = shipping.get("state", "")
            zip_code = str(shipping.get("zip", shipping.get("zip_code", "")))
            if (
                isinstance(city, str)
                and "seattle" in city.lower()
                and isinstance(state, str)
                and state.upper() in ("WA", "WASHINGTON")
                and "98101" in zip_code
            ):
                return True
    return False


AA_NESTED_OBJECTS = EvalProblem(
    category="tool_calling",
    name="aa_nested_objects",
    prompt=(
        "You have access to the following tool:\n\n"
        "```json\n"
        '{"type": "function", "function": {"name": "create_order", '
        '"description": "Create a new shipping order", '
        '"parameters": {"type": "object", "properties": {'
        '"product_id": {"type": "string", "description": "Product identifier"}, '
        '"quantity": {"type": "integer", "description": "Number of items"}, '
        '"shipping_address": {"type": "object", "properties": {'
        '"street": {"type": "string"}, '
        '"city": {"type": "string"}, '
        '"state": {"type": "string"}, '
        '"zip": {"type": "string"}, '
        '"country": {"type": "string"}}, '
        '"required": ["street", "city", "state", "zip", "country"], '
        '"description": "Shipping address object"}}, '
        '"required": ["product_id", "quantity", "shipping_address"]}}}\n'
        "```\n\n"
        "User: Order 3 units of product SKU-7890 and ship to "
        "456 Pine Ave, Seattle, WA 98101, United States.\n\n"
        "Call the tool with properly formatted arguments. "
        "Output your response as a JSON tool call."
    ),
    check=_check_aa_nested_objects,
    max_tokens=256,
)


# --- 5. Numeric coercion ---


def _check_aa_numeric_coercion(response: str) -> bool:
    """Check model passes numeric arguments as numbers, not strings."""
    calls = _parse_response_tool_calls(response)
    if not calls:
        return False
    for call in calls:
        if call.name == "transfer_funds":
            amount = call.arguments.get("amount")
            # Must be a number (int or float), close to 1500
            if isinstance(amount, (int, float)) and abs(amount - 1500.0) < 0.01:
                return True
    return False


AA_NUMERIC_COERCION = EvalProblem(
    category="tool_calling",
    name="aa_numeric_coercion",
    prompt=(
        "You have access to the following tool:\n\n"
        "```json\n"
        '{"type": "function", "function": {"name": "transfer_funds", '
        '"description": "Transfer money between accounts", '
        '"parameters": {"type": "object", "properties": {'
        '"from_account": {"type": "string", "description": "Source account ID"}, '
        '"to_account": {"type": "string", "description": "Destination account ID"}, '
        '"amount": {"type": "number", "description": "Amount to transfer (numeric, not string)"}, '
        '"currency": {"type": "string", "description": "Currency code (e.g., USD, EUR)"}}, '
        '"required": ["from_account", "to_account", "amount", "currency"]}}}\n'
        "```\n\n"
        "User: Transfer $1,500 from account ACC-001 to account ACC-002 in USD.\n\n"
        "Call the tool with properly formatted arguments. The amount must be "
        "a number, not a string. Output your response as a JSON tool call."
    ),
    check=_check_aa_numeric_coercion,
    max_tokens=256,
)


# --- 6. Boolean from natural language ---


def _check_aa_boolean_natural_lang(response: str) -> bool:
    """Check model converts natural language to boolean argument."""
    calls = _parse_response_tool_calls(response)
    if not calls:
        return False
    for call in calls:
        if call.name == "update_notification_settings":
            email_val = call.arguments.get("email_notifications")
            sms_val = call.arguments.get("sms_notifications")
            # email should be True, sms should be False
            if email_val is True and sms_val is False:
                return True
    return False


AA_BOOLEAN_NATURAL_LANG = EvalProblem(
    category="tool_calling",
    name="aa_boolean_natural_lang",
    prompt=(
        "You have access to the following tool:\n\n"
        "```json\n"
        '{"type": "function", "function": {"name": "update_notification_settings", '
        '"description": "Update user notification preferences", '
        '"parameters": {"type": "object", "properties": {'
        '"user_id": {"type": "string", "description": "User identifier"}, '
        '"email_notifications": {"type": "boolean", '
        '"description": "Whether to enable email notifications"}, '
        '"sms_notifications": {"type": "boolean", '
        '"description": "Whether to enable SMS notifications"}, '
        '"push_notifications": {"type": "boolean", '
        '"description": "Whether to enable push notifications"}}, '
        '"required": ["user_id", "email_notifications", "sms_notifications"]}}}\n'
        "```\n\n"
        "User: For user 'usr_123', turn on email notifications but "
        "disable SMS notifications.\n\n"
        "Call the tool with properly formatted arguments. "
        "Boolean values must be true/false, not strings. "
        "Output your response as a JSON tool call."
    ),
    check=_check_aa_boolean_natural_lang,
    max_tokens=256,
)


# --- 7. All required args present ---


def _check_aa_all_required_args(response: str) -> bool:
    """Check model includes ALL required arguments."""
    calls = _parse_response_tool_calls(response)
    if not calls:
        return False
    for call in calls:
        if call.name == "deploy_service":
            args = call.arguments
            required_keys = {"service_name", "version", "environment", "replicas"}
            if required_keys.issubset(set(args.keys())):
                # Validate specific values
                if (
                    args.get("service_name") == "auth-service"
                    and args.get("version") == "2.1.0"
                    and args.get("environment") == "staging"
                    and args.get("replicas") in (3, "3")
                ):
                    return True
    return False


AA_ALL_REQUIRED_ARGS = EvalProblem(
    category="tool_calling",
    name="aa_all_required_args",
    prompt=(
        "You have access to the following tool:\n\n"
        "```json\n"
        '{"type": "function", "function": {"name": "deploy_service", '
        '"description": "Deploy a service to the specified environment", '
        '"parameters": {"type": "object", "properties": {'
        '"service_name": {"type": "string", "description": "Name of the service"}, '
        '"version": {"type": "string", "description": "Version tag to deploy (e.g., 2.1.0)"}, '
        '"environment": {"type": "string", "enum": ["development", "staging", "production"], '
        '"description": "Target environment"}, '
        '"replicas": {"type": "integer", "description": "Number of replicas to run"}, '
        '"health_check_path": {"type": "string", '
        '"description": "Optional health check endpoint path"}}, '
        '"required": ["service_name", "version", "environment", "replicas"]}}}\n'
        "```\n\n"
        "User: Deploy auth-service version 2.1.0 to staging with 3 replicas.\n\n"
        "Call the tool with ALL required arguments properly formatted. "
        "Output your response as a JSON tool call."
    ),
    check=_check_aa_all_required_args,
    max_tokens=256,
)


# --- 8. Preserve exact strings ---


def _check_aa_preserve_exact_strings(response: str) -> bool:
    """Check model preserves exact string values without modification."""
    calls = _parse_response_tool_calls(response)
    if not calls:
        return False
    for call in calls:
        if call.name == "execute_command":
            command = call.arguments.get("command", "")
            # The exact command should be preserved
            if (
                isinstance(command, str)
                and "grep" in command
                and "-rn" in command
                and "TODO" in command
                and "src/" in command
            ):
                return True
    return False


AA_PRESERVE_EXACT_STRINGS = EvalProblem(
    category="tool_calling",
    name="aa_preserve_exact_strings",
    prompt=(
        "You have access to the following tool:\n\n"
        "```json\n"
        '{"type": "function", "function": {"name": "execute_command", '
        '"description": "Execute a shell command on the server", '
        '"parameters": {"type": "object", "properties": {'
        '"command": {"type": "string", '
        '"description": "The exact shell command to execute"}, '
        '"working_directory": {"type": "string", '
        '"description": "Directory to execute the command in"}, '
        '"timeout_seconds": {"type": "integer", '
        '"description": "Maximum execution time in seconds", "default": 30}}, '
        '"required": ["command"]}}}\n'
        "```\n\n"
        "User: Run this exact command: grep -rn 'TODO' src/\n\n"
        "Call the tool and pass the command string exactly as given. "
        "Do not modify, expand, or interpret the command. "
        "Output your response as a JSON tool call."
    ),
    check=_check_aa_preserve_exact_strings,
    max_tokens=256,
)


# =============================================================================
# Assembled lists by subcategory
# =============================================================================

TOOL_SELECTION_PROBLEMS: List[EvalProblem] = [
    TS_CORRECT_TOOL,
    TS_AMBIGUOUS_REQUEST,
    TS_NONE_SELECTION,
    TS_SIMILAR_NAMES,
    TS_PARAMETER_BASED,
    TS_NESTED_DESCRIPTIONS,
    TS_SPECIALIZED_VS_GENERAL,
    TS_MULTIPLE_VALID,
]

ARGUMENT_ACCURACY_PROBLEMS: List[EvalProblem] = [
    AA_DATE_ISO_FORMAT,
    AA_EMAIL_EXTRACTION,
    AA_ENUM_VALUES,
    AA_NESTED_OBJECTS,
    AA_NUMERIC_COERCION,
    AA_BOOLEAN_NATURAL_LANG,
    AA_ALL_REQUIRED_ARGS,
    AA_PRESERVE_EXACT_STRINGS,
]

# Combined list for this tier (will be extended by other tiers in future)
TOOL_CALLING_NEW_PROBLEMS: List[EvalProblem] = (
    TOOL_SELECTION_PROBLEMS + ARGUMENT_ACCURACY_PROBLEMS
)
