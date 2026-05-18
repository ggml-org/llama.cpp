/**
 * Build the conversation identity used by the server side replay buffer.
 *
 * The server identifies a stream session by a conversation id sent in the
 * X-Conversation-Id header. When the user has explicitly picked a model the
 * client appends ::modelName, which lets the router fan out direct to that
 * child for resume and stop without probing every other one. Without the
 * suffix the router falls back to a loopback probe and a DELETE fan out, both
 * still correct, just slower at the lookup step.
 */
export function streamIdentity(conversationId: string, model?: string | null): string {
	if (!conversationId) return '';
	if (!model) return conversationId;
	return `${conversationId}::${model}`;
}
