import type {
	ApiChatCompletionResponse,
	ApiChatCompletionStreamChunk,
	ApiChatCompletionToolCall,
	ApiChatCompletionToolCallDelta
} from '$lib/types/api';

export function mergeToolCallDeltas(
	existing: ApiChatCompletionToolCall[],
	deltas: ApiChatCompletionToolCallDelta[],
	indexOffset = 0
): ApiChatCompletionToolCall[] {
	const result = existing.map((call) => ({
		...call,
		function: call.function ? { ...call.function } : undefined
	}));

	for (const delta of deltas) {
		const index =
			typeof delta.index === 'number' && delta.index >= 0
				? delta.index + indexOffset
				: result.length;

		while (result.length <= index) {
			result.push({ function: undefined });
		}

		const target = result[index]!;

		if (delta.id) {
			target.id = delta.id;
		}

		if (delta.type) {
			target.type = delta.type;
		}

		if (delta.function) {
			const fn = target.function ? { ...target.function } : {};

			if (delta.function.name) {
				fn.name = delta.function.name;
			}

			if (delta.function.arguments) {
				fn.arguments = (fn.arguments ?? '') + delta.function.arguments;
			}

			target.function = fn;
		}
	}

	return result;
}

export function extractModelName(
	data: ApiChatCompletionStreamChunk | ApiChatCompletionResponse | unknown
): string | undefined {
	const asRecord = (value: unknown): Record<string, unknown> | undefined => {
		return typeof value === 'object' && value !== null
			? (value as Record<string, unknown>)
			: undefined;
	};

	const getTrimmedString = (value: unknown): string | undefined => {
		return typeof value === 'string' && value.trim() ? value.trim() : undefined;
	};

	const root = asRecord(data);
	if (!root) return undefined;

	const rootModel = getTrimmedString(root.model);
	if (rootModel) return rootModel;

	const firstChoice = Array.isArray(root.choices) ? asRecord(root.choices[0]) : undefined;
	if (!firstChoice) return undefined;

	const deltaModel = getTrimmedString(asRecord(firstChoice.delta)?.model);
	if (deltaModel) return deltaModel;

	const messageModel = getTrimmedString(asRecord(firstChoice.message)?.model);
	if (messageModel) return messageModel;

	return undefined;
}
