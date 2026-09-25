import { BuiltInTool } from '$lib/enums';

export const SANDBOX_TOOL_NAME = BuiltInTool.BROWSER_RUN_JAVASCRIPT;

export const SANDBOX_TIMEOUT_MS_DEFAULT = 10000;

export const SANDBOX_TIMEOUT_MS_MAX = 30000;

export const SANDBOX_OUTPUT_MAX_CHARS = 8192;

export const SANDBOX_EMPTY_OUTPUT = '(no output)';

export const SANDBOX_TRUNCATION_NOTICE = '[output truncated]';
