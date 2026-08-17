import { CLI_FLAGS } from './cli-flags.constants';
import { BuiltInTool, JsonSchemaType, ToolCallType } from '$lib/enums';
import type { OpenAIToolDefinition } from '$lib/types';

export const BROWSER_INFO_TOOL_NAME = BuiltInTool.GET_INFO;

/** UA token to OS name, first match wins - Android and iOS UAs also carry the Linux / Mac OS X tokens */
export const BROWSER_INFO_OS_UA_PATTERNS: readonly [RegExp, string][] = [
	[/Windows NT/, 'Windows'],
	[/Android/, 'Android'],
	[/iPhone|iPad|iPod/, 'iOS'],
	[/CrOS/, 'ChromeOS'],
	[/Mac OS X/, 'macOS'],
	[/Linux/, 'Linux']
];

export const BROWSER_INFO_OS_UNKNOWN = 'unknown';

/** Sent to the model as the `note` field of the tool result, next to the OS name */
export const BROWSER_INFO_NOTE = `This environment is browser-only, it cannot read or modify local files, and it cannot run shell commands. To get local file access, tell user to launch llama-server with the ${CLI_FLAGS.AGENT} argument.`;

export function buildBrowserInfoToolDefinition(): OpenAIToolDefinition {
	return {
		function: {
			description:
				'Get runtime info (OS name), may call when user asks about local files or shell commands',
			name: BROWSER_INFO_TOOL_NAME,
			parameters: {
				properties: {},
				required: [],
				type: JsonSchemaType.OBJECT
			}
		},
		type: ToolCallType.FUNCTION
	};
}
