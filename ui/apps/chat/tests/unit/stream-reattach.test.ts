import { MessageRole, MessageType } from '#lib/enums/index.ts';
import { ChatService } from '#lib/services/chat.service.ts';
import { DatabaseService } from '#lib/services/database.service.ts';
import { type ChatStreamHost, ChatStreamManager } from '#lib/stores/chat/streams.svelte.ts';
import { conversationsStore } from '#lib/stores/conversations/index.svelte.ts';
import type { DatabaseMessage } from '#lib/types/index.ts';
import { afterEach, expect, it, vi } from 'vitest';

vi.mock('#lib/services/database.service.ts', () => ({
	DatabaseService: { getConversationMessages: vi.fn(), updateMessage: vi.fn(async () => {}) }
}));
vi.mock('#lib/stores/conversations/index.svelte.ts', () => ({
	conversationsStore: {
		activeConversation: { id: 'conv-1' },
		findMessageIndex: () => 0,
		updateMessageAtIndex: vi.fn()
	}
}));
vi.mock('#lib/stores/models/index.svelte.ts', () => ({
	modelsStore: { selectedModelName: 'other-model' }
}));

afterEach(() => {
	vi.restoreAllMocks();
	vi.clearAllMocks();
});

it('restores the skip target while a reattached router stream is reasoning', async () => {
	const message: DatabaseMessage = {
		children: [],
		completionId: 'old-id',
		content: '',
		convId: 'conv-1',
		id: 'msg-1',
		parent: null,
		role: MessageRole.ASSISTANT,
		timestamp: 0,
		type: MessageType.TEXT
	};

	vi.mocked(DatabaseService.getConversationMessages).mockResolvedValue([{ ...message }]);
	vi.mocked(conversationsStore.updateMessageAtIndex).mockImplementation((_index, updates) => {
		Object.assign(message, updates);
	});
	vi.spyOn(ChatService, 'lookupStreamSessions').mockResolvedValue([
		{
			completed_at: 0,
			conversation_id: 'conv-1::router-child',
			is_done: false,
			started_at: 0,
			total_bytes: 0
		}
	]);
	const chunks = [
		{ choices: [{ delta: { reasoning_content: 'thinking' } }], id: 'live-id' },
		{ choices: [{ delta: { content: 'answer' } }], id: 'live-id' }
	];
	const replay =
		chunks.map((chunk) => `data: ${JSON.stringify(chunk)}\n\n`).join('') + 'data: [DONE]\n\n';

	vi.spyOn(ChatService, 'fetchStreamReplay').mockResolvedValue(new Response(replay));

	const skipTarget = vi.fn();
	const setChatReasoning = vi.fn((_convId: string, reasoning: boolean) => {
		if (reasoning) skipTarget(message.completionId, message.model);
	});
	const host = {
		activity: { isLocal: () => false },
		chatStreamingStates: new Map(),
		clearChatStreaming: vi.fn(),
		getOrCreateAbortController: () => new AbortController(),
		processing: { setActiveConversation: vi.fn(), setState: vi.fn() },
		setChatLoading: vi.fn(),
		setChatReasoning,
		setChatStreaming: vi.fn()
	} as unknown as ChatStreamHost;

	await new ChatStreamManager(host).discoverActiveStream('conv-1');
	expect(setChatReasoning.mock.calls).toEqual([
		['conv-1', true],
		['conv-1', false]
	]);
	expect(skipTarget).toHaveBeenCalledWith('live-id', 'router-child');
	expect(DatabaseService.updateMessage).toHaveBeenCalledWith('msg-1', {
		completionId: 'live-id',
		model: 'router-child'
	});
});
