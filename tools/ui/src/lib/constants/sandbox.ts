import { BuiltInTool, JsonSchemaType, ToolCallType } from '$lib/enums';
import type { OpenAIToolDefinition } from '$lib/types';

export const SANDBOX_TOOL_NAME = BuiltInTool.RUN_JAVASCRIPT;

export const SANDBOX_TIMEOUT_MS_DEFAULT = 10000;

export const SANDBOX_TIMEOUT_MS_MAX = 30000;

export const SANDBOX_OUTPUT_MAX_CHARS = 8192;

export const SANDBOX_EMPTY_OUTPUT = '(no output)';

export const SANDBOX_TRUNCATION_NOTICE = '[output truncated]';

export const SANDBOX_TOOL_DEFINITION: OpenAIToolDefinition = {
	type: ToolCallType.FUNCTION,
	function: {
		name: SANDBOX_TOOL_NAME,
		description:
			'Execute JavaScript in a sandboxed browser worker (no DOM, no page access). ' +
			'Top level await is supported. Use console.log to print intermediate values; ' +
			'a top level return statement is captured as the result.',
		parameters: {
			type: JsonSchemaType.OBJECT,
			properties: {
				code: {
					type: JsonSchemaType.STRING,
					description: 'JavaScript source to execute'
				},
				timeout_ms: {
					type: JsonSchemaType.NUMBER,
					description: `Execution timeout in milliseconds, default ${SANDBOX_TIMEOUT_MS_DEFAULT}, max ${SANDBOX_TIMEOUT_MS_MAX}`
				}
			},
			required: ['code']
		}
	}
};
