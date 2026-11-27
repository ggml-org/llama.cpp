// Plain-language lines for the tool calls of a trace. The phrasing itself is
// Llama Harness's (`createTracePhraser`), which knows llama-server's built-in
// tools; this module adds Llama Chat's own: the browser tools, web search and
// tool_search.

import {
	createTracePhraser,
	fileNameFromPath,
	toolCallPath,
	type ToolCallResolver
} from '@ggml-org/llama-harness';
import {
	TOOL_CALL_PHRASING,
	TOOL_SEARCH_LOAD_PHRASING,
	TOOL_SEARCH_PHRASING_NAME,
	TOOL_SEARCH_PHRASING_SELECT_PREFIX,
	TOOL_SEARCH_QUERY_PHRASING,
	WEB_SEARCH_PHRASING
} from '#lib/constants/index.ts';
import { BuiltInTool } from '#lib/enums/index.ts';
import { isWebSearchToolName } from '#lib/utils/index.ts';

function stringArg(args: Record<string, unknown> | null, key: string): string | undefined {
	const value = args?.[key];

	return typeof value === 'string' && value.trim() ? value : undefined;
}

const chatToolResolver: ToolCallResolver = (call) => {
	switch (call.name) {
		case BuiltInTool.BROWSER_READ_MEDIA:
			return {
				phrasing: TOOL_CALL_PHRASING[call.name],
				target: fileNameFromPath(toolCallPath(call))
			};

		case BuiltInTool.BROWSER_SEARCH_CHATS: {
			const byMatches = call.args()?.by_matches;
			const matches = Array.isArray(byMatches)
				? byMatches.filter((m): m is string => typeof m === 'string' && m.trim() !== '')
				: [];

			return { phrasing: TOOL_CALL_PHRASING[call.name], target: matches.join(', ') || undefined };
		}

		case TOOL_SEARCH_PHRASING_NAME: {
			const query = stringArg(call.args(), 'query')?.trim();

			if (query?.startsWith(TOOL_SEARCH_PHRASING_SELECT_PREFIX)) {
				const names = query
					.slice(TOOL_SEARCH_PHRASING_SELECT_PREFIX.length)
					.split(',')
					.map((name) => name.trim())
					.filter(Boolean);

				return {
					count: Math.max(names.length, 1),
					phrasing: TOOL_SEARCH_LOAD_PHRASING,
					target: names.join(', ') || undefined
				};
			}

			return { phrasing: TOOL_SEARCH_QUERY_PHRASING, target: query };
		}
	}

	if (isWebSearchToolName(call.name)) {
		return { phrasing: WEB_SEARCH_PHRASING, target: stringArg(call.args(), 'query') };
	}

	return undefined;
};

export const { describeToolActivity, describeToolCall, summarizeToolCalls } = createTracePhraser({
	phrasings: TOOL_CALL_PHRASING,
	resolvers: [chatToolResolver]
});
