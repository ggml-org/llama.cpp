/**
 * Build the conversation identity used by the server side replay buffer.
 *
 * The server identifies a stream session by a conversation id sent in the
 * X-Conversation-Id header. When the user has explicitly picked a model the
 * client appends ::modelName, so a per model session stays distinct and the
 * router resolves the owning child through its conv_id -> model map.
 */
export function streamIdentity(conversationId: string, model?: string | null): string {
	if (!conversationId) return '';
	if (!model) return conversationId;
	return `${conversationId}::${model}`;
}
