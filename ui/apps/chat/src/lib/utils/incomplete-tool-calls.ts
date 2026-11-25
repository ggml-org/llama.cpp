import { MessageRole } from '#lib/enums/index.ts';
import type { ApiChatCompletionToolCall, ApiChatMessageData } from '#lib/types/api.d.ts';

const EMPTY_ARGUMENTS = '{}';

/**
 * Drop the tool calls a stopped stream left with cut-off arguments, and the
 * tool results that answer them. The server parses every call's arguments as
 * JSON and rejects the whole request when one doesn't parse, so a call cut
 * off mid-stream would otherwise break every request after it. A call with
 * no arguments at all is kept and sent with `{}`.
 */
export function dropIncompleteToolCalls(messages: ApiChatMessageData[]): ApiChatMessageData[] {
	const droppedIds = new Set<string>();
	const result: ApiChatMessageData[] = [];

	for (const message of messages) {
		if (
			message.role === MessageRole.TOOL &&
			message.tool_call_id &&
			droppedIds.has(message.tool_call_id)
		) {
			continue;
		}

		if (!message.tool_calls?.length) {
			result.push(message);

			continue;
		}

		const calls = message.tool_calls;
		const kept: ApiChatCompletionToolCall[] = [];

		for (const call of calls) {
			const sendable = sendableToolCall(call);

			if (sendable) {
				kept.push(sendable);
			} else if (call.id) {
				droppedIds.add(call.id);
			}
		}

		if (kept.length === calls.length && kept.every((call, i) => call === calls[i])) {
			result.push(message);

			continue;
		}

		const { tool_calls: _dropped, ...rest } = message;

		result.push(kept.length > 0 ? { ...rest, tool_calls: kept } : rest);
	}

	return result;
}

/** The call as it can be sent, or null when its name or arguments were cut off. */
function sendableToolCall(call: ApiChatCompletionToolCall): ApiChatCompletionToolCall | null {
	if (!call.function?.name) return null;

	const args = call.function.arguments;

	if (!args?.trim()) {
		return { ...call, function: { ...call.function, arguments: EMPTY_ARGUMENTS } };
	}

	try {
		JSON.parse(args);

		return call;
	} catch {
		return null;
	}
}
