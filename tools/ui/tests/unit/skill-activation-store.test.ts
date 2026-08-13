import { MessageRole, MessageType } from '$lib/enums';
import { skillActivationExtra } from '$lib/services/skills-activation.service';
import { DatabaseService } from '$lib/services/database.service';
import { conversationsStore } from '$lib/stores/conversations.svelte';
import { skillActivationStore } from '$lib/stores/skill-activation.svelte';
import type { DatabaseMessage, DatabaseMessageExtraSkill } from '$lib/types';
import type { SkillBaseReadResult } from '$lib/types/skills';
import { beforeEach, describe, expect, it, vi } from 'vitest';

vi.mock('$lib/services/database.service', () => ({
	DatabaseService: {
		createMessageBranch: vi.fn(),
		createMessageBranchPair: vi.fn(),
		createRootMessage: vi.fn(),
		getConversation: vi.fn()
	}
}));

const conversationsMockState = vi.hoisted(() => ({
	activeConversation: null as { id: string; currNode: string | null } | null,
	activeMessages: [] as DatabaseMessage[],
	getConversationMessages: vi.fn()
}));

vi.mock('$lib/stores/conversations.svelte', () => ({
	conversationsStore: {
		get activeConversation() {
			return conversationsMockState.activeConversation;
		},
		get activeMessages() {
			return conversationsMockState.activeMessages;
		},
		addMessageToActive: vi.fn(),
		getConversationMessages: conversationsMockState.getConversationMessages,
		updateConversationTimestamp: vi.fn()
	}
}));

const mockCreateMessageBranch = vi.mocked(DatabaseService.createMessageBranch);
const mockCreateMessageBranchPair = vi.mocked(DatabaseService.createMessageBranchPair);
const mockGetConversationMessages = vi.mocked(conversationsStore.getConversationMessages);
const mockAddMessageToActive = vi.mocked(conversationsStore.addMessageToActive);

function baseResult(overrides: Partial<SkillBaseReadResult> = {}): SkillBaseReadResult {
	return {
		kind: 'skill',
		skill: {
			id: 'opaque-id-1',
			name: 'demo-skill',
			scope: 'project',
			provider: 'agents',
			metadata: { name: 'demo-skill', description: 'A demo skill' }
		},
		resources: { paths: [], truncated: false },
		content_xml: '<skill_content name="demo-skill">body</skill_content>',
		diagnostics: [],
		...overrides
	};
}

function resourceResult(): SkillBaseReadResult {
	return {
		kind: 'resource',
		skill: { id: 'opaque-id-1', name: 'demo-skill', scope: 'project', provider: 'agents' },
		resource: { path: 'refs/DETAILS.md' },
		content_xml: '<skill_resource>data</skill_resource>',
		diagnostics: []
	};
}

function toolResultMessage(id: string, extra: DatabaseMessageExtraSkill): DatabaseMessage {
	return {
		children: [],
		content: '<skill_content>x</skill_content>',
		convId: 'conv-1',
		extra: [extra],
		id,
		parent: 'assistant-1',
		role: MessageRole.TOOL,
		timestamp: 2,
		toolCallId: 'call_1',
		toolCalls: '',
		type: MessageType.TEXT
	} as DatabaseMessage;
}

beforeEach(() => {
	vi.clearAllMocks();
	conversationsMockState.activeConversation = null;
	conversationsMockState.activeMessages = [];
	mockGetConversationMessages.mockResolvedValue([]);
	mockCreateMessageBranch.mockImplementation((async (message) => ({
		...message,
		children: [],
		id: 'created-tool-result',
		parent: 'assistant-1'
	})) as typeof DatabaseService.createMessageBranch);
	mockCreateMessageBranchPair.mockImplementation((async (first, second) => {
		const assistant = {
			...first,
			children: ['created-tool-result'],
			id: 'created-assistant',
			parent: 'parent-1'
		};
		const toolResult = { ...second, children: [], id: 'created-tool-result', parent: 'created-assistant' };

		return [assistant, toolResult] as [DatabaseMessage, DatabaseMessage];
	}) as typeof DatabaseService.createMessageBranchPair);
});

describe('DurableSkillActivationStore (Task 4 durable seam)', () => {
	it('isActivated is false before the conversation is loaded', () => {
		expect(skillActivationStore.isActivated('conv-empty', 'opaque-id-1')).toBe(false);
	});

	it('loadConversation reconstructs durable base activations from persisted messages', async () => {
		const extra = skillActivationExtra(baseResult());

		mockGetConversationMessages.mockResolvedValue([toolResultMessage('msg-2', extra)]);

		await skillActivationStore.loadConversation('conv-reload');

		expect(skillActivationStore.isActivated('conv-reload', 'opaque-id-1')).toBe(true);
		expect(skillActivationStore.isActivated('conv-reload', 'other-id')).toBe(false);
	});

	it('loadConversation does not treat resource records as durable activations', async () => {
		mockGetConversationMessages.mockResolvedValue([
			toolResultMessage('msg-3', {
				type: 'SKILL',
				kind: 'resource',
				state: 'approved',
				name: 'demo-skill',
				scope: 'project',
				provider: 'agents',
				skillId: 'opaque-id-1',
				path: 'refs/DETAILS.md'
			} as DatabaseMessageExtraSkill)
		]);

		await skillActivationStore.loadConversation('conv-resource-only');

		expect(skillActivationStore.isActivated('conv-resource-only', 'opaque-id-1')).toBe(false);
	});

	it('loadConversation ignores malformed and unrelated extras', async () => {
		mockGetConversationMessages.mockResolvedValue([
			toolResultMessage('msg-1', {
				type: 'SKILL',
				kind: 'base',
				state: 'approved',
				name: 'demo-skill',
				scope: 'project',
				provider: 'agents',
				skillId: undefined
			} as unknown as DatabaseMessageExtraSkill)
		]);

		await skillActivationStore.loadConversation('conv-malformed');

		expect(skillActivationStore.isActivated('conv-malformed', 'opaque-id-1')).toBe(false);
	});

	it('recordActivation persists a synthetic pair for the slash path and returns the created tool result', async () => {
		conversationsMockState.activeConversation = { currNode: 'last-msg', id: 'conv-slash' };
		conversationsMockState.activeMessages = [{ id: 'last-msg', role: MessageRole.USER } as DatabaseMessage];

		const record = await skillActivationStore.recordActivation({
			conversationId: 'conv-slash',
			result: baseResult()
		});

		expect(record.created).toBe(true);
		expect(record.toolResultMessage?.id).toBe('created-tool-result');
		expect(record.extra.kind).toBe('base');
		expect(mockCreateMessageBranchPair).toHaveBeenCalledTimes(1);
		expect(mockAddMessageToActive).toHaveBeenCalled();
		expect(conversationsStore.updateConversationTimestamp).toHaveBeenCalled();

		const [assistantData, toolResultData, parentId] = mockCreateMessageBranchPair.mock.calls[0];

		expect(parentId).toBe('last-msg');
		expect(assistantData.role).toBe(MessageRole.ASSISTANT);
		expect(toolResultData.role).toBe(MessageRole.TOOL);
		expect(toolResultData.toolCallId).toBe(
			(JSON.parse(assistantData.toolCalls ?? '') as Array<{ id: string }>)[0].id
		);
		expect(toolResultData.extra).toEqual([record.extra]);
		expect(skillActivationStore.isActivated('conv-slash', 'opaque-id-1')).toBe(true);
	});

	it('recordActivation dedupes a second slash activation for the same opaque id', async () => {
		conversationsMockState.activeConversation = { currNode: 'last-msg', id: 'conv-dedupe' };
		conversationsMockState.activeMessages = [{ id: 'last-msg', role: MessageRole.USER } as DatabaseMessage];

		const first = await skillActivationStore.recordActivation({ conversationId: 'conv-dedupe', result: baseResult() });
		const second = await skillActivationStore.recordActivation({ conversationId: 'conv-dedupe', result: baseResult() });

		expect(first.created).toBe(true);
		expect(second.created).toBe(false);
		expect(second.toolResultMessage).toBeNull();
		expect(mockCreateMessageBranchPair).toHaveBeenCalledTimes(1);
	});

	it('recordActivation anchors a model read to the persisted assistant tool call carrying the model call id', async () => {
		mockGetConversationMessages.mockResolvedValue([
			{
				children: ['tool-1'],
				content: '',
				convId: 'conv-model',
				id: 'assistant-1',
				parent: 'user-1',
				role: MessageRole.ASSISTANT,
				timestamp: 1,
				toolCalls: JSON.stringify([
					{
						id: 'call_1',
						type: 'function',
						function: { name: 'read_skill', arguments: '{"name":"demo-skill"}' }
					}
				]),
				type: MessageType.TEXT
			} as DatabaseMessage
		]);

		const record = await skillActivationStore.recordActivation({
			conversationId: 'conv-model',
			result: baseResult(),
			toolCallId: 'call_1'
		});

		expect(record.created).toBe(true);
		expect(mockCreateMessageBranchPair).not.toHaveBeenCalled();
		expect(mockCreateMessageBranch).toHaveBeenCalledTimes(1);

		const [messageData, parentId] = mockCreateMessageBranch.mock.calls[0];

		expect(parentId).toBe('assistant-1');
		expect(messageData.role).toBe(MessageRole.TOOL);
		expect(messageData.toolCallId).toBe('call_1');
		expect(messageData.content).toBe('<skill_content name="demo-skill">body</skill_content>');
		expect(messageData.extra).toEqual([record.extra]);
		expect(skillActivationStore.isActivated('conv-model', 'opaque-id-1')).toBe(true);
	});

	it('recordActivation never persists a resource approval - it is session-only', async () => {
		const record = await skillActivationStore.recordActivation({
			conversationId: 'conv-resource',
			result: resourceResult()
		});

		expect(record.created).toBe(false);
		expect(record.toolResultMessage).toBeNull();
		expect(record.extra.kind).toBe('resource');
		expect(mockCreateMessageBranchPair).not.toHaveBeenCalled();
		expect(mockCreateMessageBranch).not.toHaveBeenCalled();
		// Session-scoped: authorizes the remainder of this run only.
		expect(skillActivationStore.isActivated('conv-resource', 'opaque-id-1')).toBe(true);
	});
});
