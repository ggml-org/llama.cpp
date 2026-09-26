import { BuiltInTool, JsonSchemaType, ToolCallType } from '$lib/enums';
import type { OpenAIToolDefinition } from '$lib/types';

export const GET_DATETIME_TOOL_NAME = BuiltInTool.BROWSER_GET_DATETIME;

export function buildGetDatetimeToolDefinition(): OpenAIToolDefinition {
	return {
		function: {
			description:
				'Returns the current local date and time in ISO 8601 format, with the IANA time zone name',
			name: GET_DATETIME_TOOL_NAME,
			parameters: {
				properties: {},
				required: [],
				type: JsonSchemaType.OBJECT
			}
		},
		type: ToolCallType.FUNCTION
	};
}
