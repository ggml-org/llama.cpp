// Guards durable activation persistence: record shapes, reconstruction from
// persisted messages, deduplication, and the synthetic message pair.

import { baseResult, resourceResult } from '../fixtures/skills';
import { SKILL_READ_TOOL } from '$lib/constants';
import { AttachmentType, MessageRole, MessageType } from '$lib/enums';
import { DatabaseService } from '$lib/services/database.service';
import {
	buildSkillActivationPair,
	findBaseSkillActivation,
	isBaseSkillActivation,
	isSkillExtra,
	resolveSkillSectionMeta,
	skillActivationExtra,
	skillExtraFromExtras,
	skillExtraFromMessage,
	skillResourceExtra
} from '$lib/services/skills-activation.service';
import { conversationsStore } from '$lib/stores';
import { skillActivationStore } from '$lib/stores/skill-activation.svelte';
import type { DatabaseMessage, DatabaseMessageExtra, DatabaseMessageExtraSkill } from '$lib/types';
import type {
	SkillBaseReadResult,
	SkillMetadata,
	SkillResourceReadResult
} from '$lib/types/skills';
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

vi.mock('$lib/stores/conversations/index.svelte', () => ({
	conversationsStore: {
		get activeConversation() {
			return conversationsMockState.activeConversation;
		},
		get activeMessages() {
			return conversationsMockState.activeMessages;
		},
		addMessageToActive: vi.fn(),
		getConversationMessages: conversationsMockState.getConversationMessages,
		onConversationsDeleted: vi.fn(() => () => {}),
		updateConversationTimestamp: vi.fn()
	}
}));

const mockCreateMessageBranch = vi.mocked(DatabaseService.createMessageBranch);
const mockCreateMessageBranchPair = vi.mocked(DatabaseService.createMessageBranchPair);
const mockGetConversationMessages = vi.mocked(conversationsStore.getConversationMessages);
const METADATA: SkillMetadata = { description: 'A demo skill', license: 'MIT', name: 'demo-skill' };

function activationBaseResult(): SkillBaseReadResult {
	// The durable record only carries safe server-returned fields.
	return baseResult('demo-skill', {
		content_xml: '<skill_content name="demo-skill">body &amp; more</skill_content>',
		resources: { paths: ['refs/DETAILS.md'], truncated: false },
		skill: {
			id: 'opaque-id-1',
			metadata: METADATA,
			name: 'demo-skill',
			provider: 'agents',
			scope: 'project'
		},
		source: '---\nname: demo-skill\ndescription: A demo skill\n---\n# Body'
	});
}

function activationResourceResult(): SkillResourceReadResult {
	return resourceResult('demo-skill', 'refs/DETAILS.md', {
		content_xml: '<skill_resource>data</skill_resource>',
		skill: { id: 'opaque-id-1', name: 'demo-skill', provider: 'agents', scope: 'project' }
	});
}

function toolMessage(extra?: DatabaseMessageExtraSkill, id = 'msg-1'): DatabaseMessage {
	return {
		children: [],
		content: '<skill_content>x</skill_content>',
		convId: 'conv-1',
		id,
		parent: 'assistant-1',
		role: MessageRole.TOOL,
		timestamp: 1,
		toolCallId: 'call_1',
		toolCalls: '',
		type: MessageType.TEXT,
		...(extra ? { extra: [extra] } : {})
	} as DatabaseMessage;
}

function assistantToolCallMessage(convId: string): DatabaseMessage {
	return {
		children: ['tool-1'],
		content: '',
		convId,
		id: 'assistant-1',
		parent: 'user-1',
		role: MessageRole.ASSISTANT,
		timestamp: 1,
		toolCalls: JSON.stringify([
			{
				function: { arguments: '{"name":"demo-skill"}', name: SKILL_READ_TOOL },
				id: 'call_1',
				type: 'function'
			}
		]),
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
	mockCreateMessageBranchPair.mockImplementation((async (
		first: Omit<DatabaseMessage, 'id'>,
		second: Omit<DatabaseMessage, 'id'>
	) => {
		const assistant: DatabaseMessage = {
			...first,
			children: ['created-tool-result'],
			id: 'created-assistant',
			parent: 'parent-1'
		};
		const toolResult: DatabaseMessage = {
			...second,
			children: [],
			id: 'created-tool-result',
			parent: 'created-assistant'
		};

		return [assistant, toolResult];
	}) as typeof DatabaseService.createMessageBranchPair);
});

describe('activation extras', () => {
	it('builds typed base and resource records with only safe server-returned fields', () => {
		const base = skillActivationExtra(activationBaseResult());
		const resource = skillResourceExtra(activationResourceResult());

		expect(base).toMatchObject({
			kind: 'base',
			metadata: METADATA,
			name: 'demo-skill',
			scope: 'project',
			skillId: 'opaque-id-1',
			state: 'approved',
			type: AttachmentType.SKILL
		});
		expect(base.path).toBeUndefined();
		expect(resource).toMatchObject({
			kind: 'resource',
			path: 'refs/DETAILS.md',
			skillId: 'opaque-id-1',
			type: AttachmentType.SKILL
		});
		// No content_xml, resource paths, host paths, or roots ever enter the records.
		for (const serialized of [JSON.stringify(base), JSON.stringify(resource)]) {
			expect(serialized).not.toContain('content_xml');
			expect(serialized).not.toContain('/home/');
			expect(serialized).not.toContain('cwd');
			expect('content' in base).toBe(false);
		}
	});

	it('accepts only valid records and reads the first valid SKILL extra from persisted messages', () => {
		const valid = skillActivationExtra(activationBaseResult());

		expect(isSkillExtra(valid)).toBe(true);
		expect(isSkillExtra(skillResourceExtra(activationResourceResult()))).toBe(true);
		expect(isBaseSkillActivation(valid)).toBe(true);
		expect(isBaseSkillActivation(skillResourceExtra(activationResourceResult()))).toBe(false);
		expect(isSkillExtra(null)).toBe(false);
		expect(isSkillExtra({ name: 'x', type: AttachmentType.TEXT })).toBe(false);
		expect(isSkillExtra({ ...valid, skillId: 7 })).toBe(false);
		expect(isSkillExtra({ ...valid, kind: 'other' })).toBe(false);
		expect(isSkillExtra({ ...valid, state: 'denied' })).toBe(false);
		expect(isSkillExtra({ ...valid, scope: 'host' })).toBe(false);

		expect(
			skillExtraFromExtras([
				{ content: 'c', name: 't', type: AttachmentType.TEXT },
				{ ...valid, skillId: undefined } as unknown as DatabaseMessageExtra,
				valid
			])
		).toEqual(valid);
		expect(skillExtraFromExtras(undefined)).toBeUndefined();
		expect(skillExtraFromMessage(toolMessage(valid))).toEqual(valid);
		expect(skillExtraFromMessage(toolMessage())).toBeUndefined();
	});
});

describe('reconstruction from persisted messages', () => {
	it('finds the durable base activation by exact opaque id, ignoring resources and malformed extras', () => {
		const extra = skillActivationExtra(activationBaseResult());
		const messages = [
			toolMessage(),
			toolMessage(extra, 'msg-2'),
			toolMessage(skillResourceExtra(activationResourceResult()), 'msg-3'),
			toolMessage(
				{
					...skillActivationExtra(activationBaseResult()),
					skillId: undefined
				} as unknown as DatabaseMessageExtraSkill,
				'msg-4'
			)
		];

		expect(findBaseSkillActivation(messages, 'opaque-id-1')?.id).toBe('msg-2');
		expect(findBaseSkillActivation(messages, 'other-id')).toBeUndefined();
		expect(findBaseSkillActivation([], 'opaque-id-1')).toBeUndefined();
	});

	it('resolves safe display metadata for rendering and falls back for unknown tools or bad metadata', () => {
		expect(
			resolveSkillSectionMeta({
				toolName: SKILL_READ_TOOL,
				toolResultExtras: [skillActivationExtra(activationBaseResult())]
			})
		).toEqual({ kind: 'base', name: 'demo-skill', provider: 'agents', scope: 'project' });
		expect(
			resolveSkillSectionMeta({
				toolName: SKILL_READ_TOOL,
				toolResultExtras: [skillResourceExtra(activationResourceResult())]
			})
		).toEqual({
			kind: 'resource',
			name: 'demo-skill',
			path: 'refs/DETAILS.md',
			provider: 'agents',
			scope: 'project'
		});
		expect(
			resolveSkillSectionMeta({
				toolName: 'other_tool',
				toolResultExtras: [skillActivationExtra(activationBaseResult())]
			})
		).toBeUndefined();
		expect(resolveSkillSectionMeta({ toolName: SKILL_READ_TOOL })).toBeUndefined();
	});
});

describe('buildSkillActivationPair', () => {
	it('builds a valid synthetic assistant tool-call pair with typed metadata and no host paths', () => {
		const pair = buildSkillActivationPair(activationBaseResult(), {
			conversationId: 'conv-1',
			cwd: '/home/user'
		});
		const calls = JSON.parse(pair.assistant.toolCalls ?? '') as Array<{
			id: string;
			function: { name: string; arguments: string };
		}>;

		expect(pair.assistant.role).toBe(MessageRole.ASSISTANT);
		expect(pair.assistant.content).toBe('');
		expect(calls).toHaveLength(1);
		expect(calls[0].function.name).toBe(SKILL_READ_TOOL);
		expect(JSON.parse(calls[0].function.arguments)).toEqual({ name: 'demo-skill' });
		expect(pair.toolResult.role).toBe(MessageRole.TOOL);
		expect(pair.toolResult.toolCallId).toBe(calls[0].id);
		expect(pair.toolResult.content).toBe(
			'<skill_content name="demo-skill">body &amp; more</skill_content>'
		);
		expect(pair.toolResult.extra?.[0]).toMatchObject({ kind: 'base', skillId: 'opaque-id-1' });
		expect(pair.toolResult.toolCwd).toBeUndefined();
		expect(JSON.stringify(pair)).not.toContain('/home/user');
		expect(JSON.stringify(pair)).not.toContain('refs/DETAILS.md');
	});
});

describe('DurableSkillActivationStore', () => {
	it('loadConversation reconstructs durable base activations only, ignoring resource and malformed records', async () => {
		mockGetConversationMessages.mockResolvedValue([
			toolMessage(skillActivationExtra(activationBaseResult()), 'msg-2'),
			toolMessage(skillResourceExtra(activationResourceResult()), 'msg-3')
		]);

		await skillActivationStore.loadConversation('conv-reload');

		expect(skillActivationStore.isActivated('conv-reload', 'opaque-id-1')).toBe(true);
		expect(skillActivationStore.isActivated('conv-reload', 'other-id')).toBe(false);
		expect(skillActivationStore.isActivated('conv-unloaded', 'opaque-id-1')).toBe(false);
	});

	it('recordActivation persists a synthetic pair for the slash path and returns the created tool result', async () => {
		conversationsMockState.activeConversation = { currNode: 'last-msg', id: 'conv-slash' };
		conversationsMockState.activeMessages = [
			{ id: 'last-msg', role: MessageRole.USER } as DatabaseMessage
		];

		const record = await skillActivationStore.recordActivation({
			conversationId: 'conv-slash',
			result: activationBaseResult()
		});

		expect(record.created).toBe(true);
		expect(record.toolResultMessage?.id).toBe('created-tool-result');
		expect(record.extra.kind).toBe('base');

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

	it('recordActivation dedupes concurrent slash and model activations of the same opaque id into one durable record', async () => {
		const convId = 'conv-race';

		conversationsMockState.activeConversation = { currNode: 'last-msg', id: convId };
		conversationsMockState.activeMessages = [
			{ id: 'last-msg', role: MessageRole.USER } as DatabaseMessage
		];
		mockGetConversationMessages.mockResolvedValue([assistantToolCallMessage(convId)]);

		// Concurrent slash and model-driven activations share one in-flight
		// persistence transaction; the joining call reuses it.
		const [slashRecord, modelRecord] = await Promise.all([
			skillActivationStore.recordActivation({
				conversationId: convId,
				result: activationBaseResult()
			}),
			skillActivationStore.recordActivation({
				conversationId: convId,
				result: activationBaseResult(),
				toolCallId: 'call_1'
			})
		]);

		expect(slashRecord.created).toBe(true);
		expect(modelRecord.created).toBe(false);
		expect(modelRecord.toolResultMessage).toBeNull();
		expect(mockCreateMessageBranchPair).toHaveBeenCalledTimes(1);
		expect(skillActivationStore.isActivated(convId, 'opaque-id-1')).toBe(true);
	});

	it('recordActivation anchors a model read to the persisted assistant tool call carrying the model call id', async () => {
		mockGetConversationMessages.mockResolvedValue([assistantToolCallMessage('conv-model')]);

		const record = await skillActivationStore.recordActivation({
			conversationId: 'conv-model',
			result: activationBaseResult(),
			toolCallId: 'call_1'
		});

		expect(record.created).toBe(true);
		expect(mockCreateMessageBranchPair).not.toHaveBeenCalled();

		const [messageData, parentId] = mockCreateMessageBranch.mock.calls[0];

		expect(parentId).toBe('assistant-1');
		expect(messageData.role).toBe(MessageRole.TOOL);
		expect(messageData.toolCallId).toBe('call_1');
		expect(messageData.content).toBe(
			'<skill_content name="demo-skill">body &amp; more</skill_content>'
		);
		expect(messageData.extra).toEqual([record.extra]);
	});

	it('never persists a resource approval and clears the in-flight slot on a failed persistence', async () => {
		const resource = await skillActivationStore.recordActivation({
			conversationId: 'conv-resource',
			result: activationResourceResult()
		});

		expect(resource.created).toBe(false);
		expect(resource.toolResultMessage).toBeNull();
		expect(mockCreateMessageBranchPair).not.toHaveBeenCalled();
		// Session-scoped: authorizes the remainder of this run only.
		expect(skillActivationStore.isActivated('conv-resource', 'opaque-id-1')).toBe(true);

		conversationsMockState.activeConversation = { currNode: 'last-msg', id: 'conv-fail' };
		conversationsMockState.activeMessages = [
			{ id: 'last-msg', role: MessageRole.USER } as DatabaseMessage
		];
		mockCreateMessageBranchPair.mockRejectedValueOnce(new Error('db write failed'));

		const first = skillActivationStore.recordActivation({
			conversationId: 'conv-fail',
			result: activationBaseResult()
		});
		const second = skillActivationStore.recordActivation({
			conversationId: 'conv-fail',
			result: activationBaseResult()
		});

		await expect(first).rejects.toThrow('db write failed');
		await expect(second).rejects.toThrow('db write failed');
		expect(skillActivationStore.isActivated('conv-fail', 'opaque-id-1')).toBe(false);

		// The failed transaction is not sticky: a later activation retries.
		const retry = await skillActivationStore.recordActivation({
			conversationId: 'conv-fail',
			result: activationBaseResult()
		});

		expect(retry.created).toBe(true);
		expect(skillActivationStore.isActivated('conv-fail', 'opaque-id-1')).toBe(true);
	});
});
