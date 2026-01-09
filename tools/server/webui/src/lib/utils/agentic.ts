import type { ApiChatMessageData } from '$lib/types/api';
import type { AgenticMessage } from '$lib/types/agentic';

/**
 * Converts API messages to agentic format.
 */
export function toAgenticMessages(messages: ApiChatMessageData[]): AgenticMessage[] {
	return messages.map((message) => {
		if (message.role === 'assistant' && message.tool_calls && message.tool_calls.length > 0) {
			return {
				role: 'assistant',
				content: message.content,
				tool_calls: message.tool_calls.map((call, index) => ({
					id: call.id ?? `call_${index}`,
					type: (call.type as 'function') ?? 'function',
					function: {
						name: call.function?.name ?? '',
						arguments: call.function?.arguments ?? ''
					}
				}))
			} satisfies AgenticMessage;
		}

		if (message.role === 'tool' && message.tool_call_id) {
			return {
				role: 'tool',
				tool_call_id: message.tool_call_id,
				content: typeof message.content === 'string' ? message.content : ''
			} satisfies AgenticMessage;
		}

		return {
			role: message.role,
			content: message.content
		} satisfies AgenticMessage;
	});
}
