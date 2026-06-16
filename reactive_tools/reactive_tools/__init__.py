"""reactive_tools — in-process reactive event plane + tool layer.

Workspace member (d11) resolving into the single shared root .venv so it runs
in the SAME interpreter as llm_framework (d2 — one in-process process).

Public surface
--------------
- :class:`Event`        — an immutable ``(kind, payload)`` event carrier
- :class:`Subscription` — an async-iterable subscriber handle
- :class:`EventPlane`   — the in-process publish/subscribe bus
"""

from __future__ import annotations

from .event_plane import Event, EventPlane, Subscription
from .tool_hook import (
    EVENT_TOOL_CALL,
    EVENT_TOOL_RESULT,
    ToolError,
    ToolHook,
    ToolRegistry,
    ToolResult,
    ToolSpec,
    build_default_hook,
)
from .tools import ToolInputError, register_core_tools
from .tool_registry import (
    AGENTIC_TOOL_NAMES,
    DEFAULT_SELECT_MAX_TOKENS,
    ECHO_TOOL,
    EchoArgs,
    GrowableToolRegistry,
    StructuredToolCaller,
    ToolCallResult,
    ToolDef,
    ToolRegistryError,
    ToolRuntime,
    ToolSelection,
    build_tool_runtime,
    register_agentic_tools,
)
from .email_tool import make_send_email, register_email_tool
from .web_tools import (
    WEB_FETCH_TOOL,
    WEB_SEARCH_TOOL,
    ResultCache,
    WebFetchArgs,
    WebSearchArgs,
    ddgs_backend,
    make_web_fetch,
    make_web_search,
    register_web_tools,
)
from .send_mail_tool import (
    MailAdapter,
    SendMailArgs,
    SmtpAppPasswordAdapter,
    make_send_mail_handler,
    make_send_mail_tool,
    register_send_mail,
    SEND_MAIL_NAME,
    SEND_MAIL_DESCRIPTION,
)
from .subscriptions import (
    DATA_PLANE_KINDS,
    META_LAMBDA_CLOSED,
    META_LAMBDA_FIRED,
    META_LAMBDA_OBSERVATION,
    META_LAMBDA_REGISTERED,
    LambdaInputError,
    LambdaRecord,
    LambdaRegistry,
    build_reducer,
)
from .file_tools import (
    DEFAULT_WORKSPACE_ROOT,
    WORKSPACE_ROOT_ENV,
    FileReadArgs,
    FileWriteArgs,
    build_filesystem_tools,
    make_file_read,
    make_file_write,
    register_filesystem_tools,
    resolve_workspace_root,
)
from .lambda_tools import register_lambda_tools
from .cron_store import (
    CronJob,
    CronStore,
    DB_FILENAME as CRON_DB_FILENAME,
    ParsedCron,
    cron_matches,
    iter_due_fire_times,
    next_fire_after,
    parse_cron_expression,
    resolve_data_dir as resolve_cron_data_dir,
    resolve_db_path as resolve_cron_db_path,
    validate_cron_expression,
)
from .cron_tools import (
    CRON_ADD_NAME,
    CRON_DELETE_NAME,
    CRON_LIST_NAME,
    CronAddArgs,
    CronDeleteArgs,
    CronListArgs,
    build_cron_tools,
    make_cron_add,
    make_cron_delete,
    make_cron_list,
    register_cron_tools,
)
from .scheduler import (
    EVENT_JOB_CANCELLED,
    EVENT_JOB_COMPLETED,
    EVENT_JOB_ERROR,
    EVENT_JOB_FIRED,
    EVENT_JOB_SCHEDULED,
    KIND_INTERVAL,
    KIND_ONE_SHOT,
    SCHEDULER_EVENT_KINDS,
    ScheduledJob,
    Scheduler,
)

__all__ = [
    # event plane (a1)
    "Event",
    "EventPlane",
    "Subscription",
    # tool hook + registry (a2)
    "ToolHook",
    "ToolRegistry",
    "ToolResult",
    "ToolSpec",
    "ToolError",
    "ToolInputError",
    "build_default_hook",
    "register_core_tools",
    # growable Pydantic-typed tool registry + native structured tool-calling (s2 a1)
    "ToolDef",
    "GrowableToolRegistry",
    "StructuredToolCaller",
    "ToolSelection",
    "ToolCallResult",
    "ToolRuntime",
    "ToolRegistryError",
    "EchoArgs",
    "ECHO_TOOL",
    "build_tool_runtime",
    "DEFAULT_SELECT_MAX_TOKENS",
    # the SIX s2 node→tool surface composer + its canonical name set (s3/b5)
    "AGENTIC_TOOL_NAMES",
    "register_agentic_tools",
    # email delivery channel (s5 a2)
    "register_email_tool",
    "make_send_email",
    # web tools — ddgs search + Trafilatura markdown fetch (s2 a2)
    "WEB_SEARCH_TOOL",
    "WEB_FETCH_TOOL",
    "WebSearchArgs",
    "WebFetchArgs",
    "ResultCache",
    "ddgs_backend",
    "make_web_search",
    "make_web_fetch",
    "register_web_tools",
    # send_mail agentic tool — one ToolDef, swappable adapter, recipient-locked (s2 a4)
    "MailAdapter",
    "SendMailArgs",
    "SmtpAppPasswordAdapter",
    "make_send_mail_handler",
    "make_send_mail_tool",
    "register_send_mail",
    "SEND_MAIL_NAME",
    "SEND_MAIL_DESCRIPTION",
    # filesystem tools — file_read + hard-sandboxed file_write, one ToolDef each (s2 a3)
    "FileReadArgs",
    "FileWriteArgs",
    "make_file_read",
    "make_file_write",
    "build_filesystem_tools",
    "register_filesystem_tools",
    "resolve_workspace_root",
    "WORKSPACE_ROOT_ENV",
    "DEFAULT_WORKSPACE_ROOT",
    "EVENT_TOOL_CALL",
    "EVENT_TOOL_RESULT",
    # reactive lambdas at scale + read-only live-subscriptions surface (s1 b2)
    "LambdaRegistry",
    "LambdaRecord",
    "LambdaInputError",
    "build_reducer",
    "register_lambda_tools",
    "DATA_PLANE_KINDS",
    "META_LAMBDA_REGISTERED",
    "META_LAMBDA_FIRED",
    "META_LAMBDA_CLOSED",
    "META_LAMBDA_OBSERVATION",
    # in-process scheduler (s5 a3) — fires recurring/one-shot workflow specs
    "Scheduler",
    "ScheduledJob",
    "KIND_INTERVAL",
    "KIND_ONE_SHOT",
    "SCHEDULER_EVENT_KINDS",
    "EVENT_JOB_SCHEDULED",
    "EVENT_JOB_FIRED",
    "EVENT_JOB_COMPLETED",
    "EVENT_JOB_ERROR",
    "EVENT_JOB_CANCELLED",
    # cron tools — cron_add/cron_list/cron_delete, one ToolDef each; shared-SQLite
    # persistence (firing scheduler is s6) (s2 a5)
    "CronStore",
    "CronJob",
    "CRON_DB_FILENAME",
    "ParsedCron",
    "parse_cron_expression",
    "cron_matches",
    "iter_due_fire_times",
    "next_fire_after",
    "resolve_cron_data_dir",
    "resolve_cron_db_path",
    "validate_cron_expression",
    "CRON_ADD_NAME",
    "CRON_LIST_NAME",
    "CRON_DELETE_NAME",
    "CronAddArgs",
    "CronListArgs",
    "CronDeleteArgs",
    "make_cron_add",
    "make_cron_list",
    "make_cron_delete",
    "build_cron_tools",
    "register_cron_tools",
]
