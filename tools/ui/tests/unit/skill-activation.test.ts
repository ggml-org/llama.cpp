import { SKILL_READ_TOOL } from '$lib/constants';
import { AttachmentType, MessageRole, MessageType } from '$lib/enums';
import { DatabaseService } from '$lib/services/database.service';
import { SkillsService } from '$lib/services/skills.service';
import {
	buildSkillActivationPair,
	findBaseSkillActivation,
	isBaseSkillActivation,
	isSkillExtra,
	isSkillToolSection,
	resolveSkillSectionMeta,
	skillActivationExtra,
	skillExtraFromExtras,
	skillExtraFromMessage,
	skillResourceExtra
} from '$lib/services/skills-activation.service';
import { conversationsStore } from '$lib/stores/conversations.svelte';
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

const METADATA: SkillMetadata = {
	description: 'A demo skill',
	license: 'MIT',
	name: 'demo-skill'
};

function baseResult(overrides: Partial<SkillBaseReadResult> = {}): SkillBaseReadResult {
	return {
		body_markdown: '# Body',
		content_xml: '<skill_content name="demo-skill">body &amp; more</skill_content>',
		diagnostics: [],
		kind: 'skill',
		resources: { paths: ['refs/DETAILS.md'], truncated: false },
		skill: {
			id: 'opaque-id-1',
			metadata: METADATA,
			name: 'demo-skill',
			provider: 'agents',
			scope: 'project'
		},
		source: '---\nname: demo-skill\ndescription: A demo skill\n---\n# Body',
		...overrides
	};
}

function resourceResult(overrides: Partial<SkillResourceReadResult> = {}): SkillResourceReadResult {
	return {
		content_xml: '<skill_resource>data</skill_resource>',
		diagnostics: [],
		kind: 'resource',
		resource: { path: 'refs/DETAILS.md' },
		source: 'data',
		skill: { id: 'opaque-id-1', name: 'demo-skill', provider: 'agents', scope: 'project' },
		...overrides
	};
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
				function: { arguments: '{"name":"demo-skill"}', name: 'read_skill' },
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

describe('skillActivationExtra', () => {
	it('builds a typed durable base record with only safe server-returned fields', () => {
		const extra = skillActivationExtra(baseResult());

		expect(extra.type).toBe(AttachmentType.SKILL);
		expect(extra.kind).toBe('base');
		expect(extra.state).toBe('approved');
		expect(extra.skillId).toBe('opaque-id-1');
		expect(extra.name).toBe('demo-skill');
		expect(extra.scope).toBe('project');
		expect(extra.provider).toBe('agents');
		expect(extra.metadata).toEqual(METADATA);
		expect(extra.path).toBeUndefined();
		// No content_xml, resource paths, host paths, or roots ever enter the record.
		expect(JSON.stringify(extra)).not.toContain('content_xml');
		expect(JSON.stringify(extra)).not.toContain('refs/DETAILS.md');
		expect(JSON.stringify(extra)).not.toContain('/home/');
		expect(JSON.stringify(extra)).not.toContain('cwd');
		expect('content' in extra).toBe(false);
	});

	it('omits the metadata field when the server returned none', () => {
		const extra = skillActivationExtra(
			baseResult({ skill: { id: 'x', name: 'n', provider: 'p', scope: 'global' } })
		);

		expect(extra.metadata).toBeUndefined();
	});
});

describe('skillResourceExtra', () => {
	it('builds a resource record carrying the requested relative path', () => {
		const extra = skillResourceExtra(resourceResult());

		expect(extra.type).toBe(AttachmentType.SKILL);
		expect(extra.kind).toBe('resource');
		expect(extra.state).toBe('approved');
		expect(extra.skillId).toBe('opaque-id-1');
		expect(extra.path).toBe('refs/DETAILS.md');
		expect(extra.metadata).toBeUndefined();
		expect(JSON.stringify(extra)).not.toContain('content_xml');
	});
});

describe('isSkillExtra', () => {
	it('accepts valid base and resource records', () => {
		expect(isSkillExtra(skillActivationExtra(baseResult()))).toBe(true);
		expect(isSkillExtra(skillResourceExtra(resourceResult()))).toBe(true);
	});

	it('rejects non-SKILL extras and malformed SKILL-shaped records', () => {
		expect(isSkillExtra(null)).toBe(false);
		expect(isSkillExtra(undefined)).toBe(false);
		expect(isSkillExtra('nope')).toBe(false);
		expect(isSkillExtra(42)).toBe(false);
		expect(isSkillExtra({ name: 'x', type: AttachmentType.TEXT })).toBe(false);

		const valid = skillActivationExtra(baseResult());
		const { skillId: _skillId, ...missingId } = valid;

		expect(isSkillExtra({ ...valid, skillId: undefined })).toBe(false);
		expect(isSkillExtra({ ...valid, skillId: 7 })).toBe(false);
		expect(isSkillExtra({ ...valid, kind: 'other' })).toBe(false);
		expect(isSkillExtra({ ...valid, state: 'denied' })).toBe(false);
		expect(isSkillExtra({ ...valid, name: '' })).toBe(false);
		expect(isSkillExtra({ ...valid, scope: 'host' })).toBe(false);
		expect(isSkillExtra(missingId)).toBe(false);
	});
});

describe('skillExtraFromExtras / skillExtraFromMessage', () => {
	it('returns the first valid SKILL extra, ignoring unrelated and malformed entries', () => {
		const valid = skillActivationExtra(baseResult());

		expect(
			skillExtraFromExtras([
				{ content: 'c', name: 't', type: AttachmentType.TEXT },
				{ ...valid, skillId: undefined } as unknown as DatabaseMessageExtra,
				valid
			])
		).toEqual(valid);
	});

	it('returns undefined for empty, undefined, or all-invalid extras', () => {
		expect(skillExtraFromExtras(undefined)).toBeUndefined();
		expect(skillExtraFromExtras([])).toBeUndefined();
		expect(
			skillExtraFromExtras([
				{ base64Url: 'data:image/png;base64,AA==', name: 'i', type: AttachmentType.IMAGE }
			])
		).toBeUndefined();
	});

	it('reads the SKILL extra from a persisted message', () => {
		const valid = skillActivationExtra(baseResult());

		expect(skillExtraFromMessage(toolMessage(valid))).toEqual(valid);
		expect(skillExtraFromMessage(toolMessage())).toBeUndefined();
	});
});

describe('isBaseSkillActivation', () => {
	it('distinguishes durable base records from resource records', () => {
		expect(isBaseSkillActivation(skillActivationExtra(baseResult()))).toBe(true);
		expect(isBaseSkillActivation(skillResourceExtra(resourceResult()))).toBe(false);
	});
});

describe('findBaseSkillActivation', () => {
	it('reconstructs an activated identity by its exact opaque id from persisted tool messages', () => {
		const extra = skillActivationExtra(baseResult());
		const messages = [
			toolMessage(),
			toolMessage(extra, 'msg-2'),
			toolMessage(skillResourceExtra(resourceResult()), 'msg-3')
		];

		expect(findBaseSkillActivation(messages, 'opaque-id-1')?.id).toBe('msg-2');
	});

	it('ignores resource records and malformed extras for reconstruction', () => {
		const messages = [
			toolMessage(skillResourceExtra(resourceResult())),
			toolMessage({
				...skillActivationExtra(baseResult()),
				skillId: undefined
			} as unknown as DatabaseMessageExtraSkill)
		];

		expect(findBaseSkillActivation(messages, 'opaque-id-1')).toBeUndefined();
	});

	it('returns undefined when the opaque id was never activated', () => {
		expect(
			findBaseSkillActivation([toolMessage(skillActivationExtra(baseResult()))], 'other-id')
		).toBeUndefined();
		expect(findBaseSkillActivation([], 'opaque-id-1')).toBeUndefined();
	});
});

describe('buildSkillActivationPair', () => {
	it('builds a valid synthetic assistant tool-call message paired with its tool result', () => {
		const pair = buildSkillActivationPair(baseResult(), {
			conversationId: 'conv-1',
			cwd: undefined
		});

		expect(pair.assistant.role).toBe(MessageRole.ASSISTANT);
		expect(pair.assistant.convId).toBe('conv-1');
		expect(pair.assistant.content).toBe('');
		expect(pair.assistant.type).toBe(MessageType.TEXT);
		expect(pair.toolResult.role).toBe(MessageRole.TOOL);
		expect(pair.toolResult.convId).toBe('conv-1');
		expect(pair.toolResult.content).toBe(
			'<skill_content name="demo-skill">body &amp; more</skill_content>'
		);

		const calls = JSON.parse(pair.assistant.toolCalls ?? '') as Array<{
			id: string;
			type: string;
			function: { name: string; arguments: string };
		}>;

		expect(calls).toHaveLength(1);
		expect(calls[0].function.name).toBe(SKILL_READ_TOOL);
		expect(JSON.parse(calls[0].function.arguments)).toEqual({ name: 'demo-skill' });
		expect(pair.toolResult.toolCallId).toBe(calls[0].id);
	});

	it('records the typed base activation metadata on the paired tool result and never carries host paths', () => {
		const pair = buildSkillActivationPair(baseResult(), {
			conversationId: 'conv-1',
			cwd: '/home/user'
		});
		const [extra] = pair.toolResult.extra ?? [];

		expect(isSkillExtra(extra)).toBe(true);

		if (isSkillExtra(extra)) {
			expect(extra.kind).toBe('base');
			expect(extra.skillId).toBe('opaque-id-1');
		}

		expect(pair.toolResult.toolCwd).toBeUndefined();
		expect(JSON.stringify(pair)).not.toContain('content_xml');
		expect(JSON.stringify(pair)).not.toContain('/home/user');
		expect(JSON.stringify(pair)).not.toContain('refs/DETAILS.md');
	});
});

describe('resolveSkillSectionMeta / isSkillToolSection', () => {
	it('resolves safe display metadata from a valid base record', () => {
		const section = {
			toolName: SKILL_READ_TOOL,
			toolResultExtras: [skillActivationExtra(baseResult())]
		};

		expect(resolveSkillSectionMeta(section)).toEqual({
			kind: 'base',
			name: 'demo-skill',
			provider: 'agents',
			scope: 'project'
		});
		expect(isSkillToolSection(section)).toBe(true);
	});

	it('resolves the requested resource path from a resource record', () => {
		const section = {
			toolName: SKILL_READ_TOOL,
			toolResultExtras: [skillResourceExtra(resourceResult())]
		};

		expect(resolveSkillSectionMeta(section)).toEqual({
			kind: 'resource',
			name: 'demo-skill',
			path: 'refs/DETAILS.md',
			provider: 'agents',
			scope: 'project'
		});
	});

	it('falls back to generic rendering for unknown tools, missing, or malformed metadata', () => {
		expect(
			resolveSkillSectionMeta({
				toolName: 'other_tool',
				toolResultExtras: [skillActivationExtra(baseResult())]
			})
		).toBeUndefined();
		expect(
			resolveSkillSectionMeta({ toolName: SKILL_READ_TOOL, toolResultExtras: [] })
		).toBeUndefined();
		expect(
			resolveSkillSectionMeta({
				toolName: SKILL_READ_TOOL,
				toolResultExtras: [{ content: 'x', name: 'x', type: AttachmentType.TEXT }]
			})
		).toBeUndefined();
		expect(resolveSkillSectionMeta({ toolName: SKILL_READ_TOOL })).toBeUndefined();
		expect(isSkillToolSection({ toolName: SKILL_READ_TOOL })).toBe(false);
	});
});

describe('DurableSkillActivationStore (Task 4 durable seam)', () => {
	it('isActivated is false before the conversation is loaded', () => {
		expect(skillActivationStore.isActivated('conv-empty', 'opaque-id-1')).toBe(false);
	});

	it('loadConversation reconstructs durable base activations from persisted messages', async () => {
		const extra = skillActivationExtra(baseResult());

		mockGetConversationMessages.mockResolvedValue([toolMessage(extra, 'msg-2')]);

		await skillActivationStore.loadConversation('conv-reload');

		expect(skillActivationStore.isActivated('conv-reload', 'opaque-id-1')).toBe(true);
		expect(skillActivationStore.isActivated('conv-reload', 'other-id')).toBe(false);
	});

	it('loadConversation does not treat resource records as durable activations', async () => {
		mockGetConversationMessages.mockResolvedValue([
			toolMessage(
				{
					kind: 'resource',
					name: 'demo-skill',
					path: 'refs/DETAILS.md',
					provider: 'agents',
					scope: 'project',
					skillId: 'opaque-id-1',
					state: 'approved',
					type: 'SKILL'
				} as DatabaseMessageExtraSkill,
				'msg-3'
			)
		]);

		await skillActivationStore.loadConversation('conv-resource-only');

		expect(skillActivationStore.isActivated('conv-resource-only', 'opaque-id-1')).toBe(false);
	});

	it('loadConversation ignores malformed and unrelated extras', async () => {
		mockGetConversationMessages.mockResolvedValue([
			toolMessage(
				{
					kind: 'base',
					name: 'demo-skill',
					provider: 'agents',
					scope: 'project',
					skillId: undefined,
					state: 'approved',
					type: 'SKILL'
				} as unknown as DatabaseMessageExtraSkill
			)
		]);

		await skillActivationStore.loadConversation('conv-malformed');

		expect(skillActivationStore.isActivated('conv-malformed', 'opaque-id-1')).toBe(false);
	});

	it('recordActivation persists a synthetic pair for the slash path and returns the created tool result', async () => {
		conversationsMockState.activeConversation = { currNode: 'last-msg', id: 'conv-slash' };
		conversationsMockState.activeMessages = [
			{ id: 'last-msg', role: MessageRole.USER } as DatabaseMessage
		];

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
		conversationsMockState.activeMessages = [
			{ id: 'last-msg', role: MessageRole.USER } as DatabaseMessage
		];

		const first = await skillActivationStore.recordActivation({
			conversationId: 'conv-dedupe',
			result: baseResult()
		});
		const second = await skillActivationStore.recordActivation({
			conversationId: 'conv-dedupe',
			result: baseResult()
		});

		expect(first.created).toBe(true);
		expect(second.created).toBe(false);
		expect(second.toolResultMessage).toBeNull();
		expect(mockCreateMessageBranchPair).toHaveBeenCalledTimes(1);
	});

	it.each(['slash-first', 'model-first'] as const)(
		'recordActivation serializes concurrent slash and model activations of the same opaque id into one durable record (%s)',
		async (order) => {
			const convId = order === 'slash-first' ? 'conv-cross-race' : 'conv-cross-race-2';

			conversationsMockState.activeConversation = { currNode: 'last-msg', id: convId };
			conversationsMockState.activeMessages = [
				{ id: 'last-msg', role: MessageRole.USER } as DatabaseMessage
			];

			mockGetConversationMessages.mockResolvedValue([assistantToolCallMessage(convId)]);

			// Concurrent calls must share one in-flight persistence transaction.
			const firstCall =
				order === 'slash-first'
					? skillActivationStore.recordActivation({
							conversationId: convId,
							result: baseResult()
						})
					: skillActivationStore.recordActivation({
							conversationId: convId,
							result: baseResult(),
							toolCallId: 'call_1'
						});
			const secondCall =
				order === 'slash-first'
					? skillActivationStore.recordActivation({
							conversationId: convId,
							result: baseResult(),
							toolCallId: 'call_1'
						})
					: skillActivationStore.recordActivation({
							conversationId: convId,
							result: baseResult()
						});
			const [firstRecord, secondRecord] = await Promise.all([firstCall, secondCall]);

			// The joining call reuses the first persisted activation.
			expect(firstRecord.created).toBe(true);
			expect(secondRecord.created).toBe(false);
			expect(secondRecord.toolResultMessage).toBeNull();
			expect(secondRecord.extra.skillId).toBe('opaque-id-1');
			if (order === 'slash-first') {
				expect(mockCreateMessageBranch).not.toHaveBeenCalled();
				expect(mockCreateMessageBranchPair).toHaveBeenCalledTimes(1);
			} else {
				expect(mockCreateMessageBranchPair).not.toHaveBeenCalled();
				expect(mockCreateMessageBranch).toHaveBeenCalledTimes(1);
			}
			expect(skillActivationStore.isActivated(convId, 'opaque-id-1')).toBe(true);
		}
	);

	it('recordActivation persists nothing and clears the in-flight slot when the persistence fails', async () => {
		conversationsMockState.activeConversation = { currNode: 'last-msg', id: 'conv-fail-retry' };
		conversationsMockState.activeMessages = [
			{ id: 'last-msg', role: MessageRole.USER } as DatabaseMessage
		];

		mockCreateMessageBranchPair.mockRejectedValueOnce(new Error('db write failed'));

		const first = skillActivationStore.recordActivation({
			conversationId: 'conv-fail-retry',
			result: baseResult()
		});
		const second = skillActivationStore.recordActivation({
			conversationId: 'conv-fail-retry',
			result: baseResult()
		});

		await expect(first).rejects.toThrow('db write failed');
		await expect(second).rejects.toThrow('db write failed');
		// No durable activation on a failed persistence.
		expect(skillActivationStore.isActivated('conv-fail-retry', 'opaque-id-1')).toBe(false);

		// The failed transaction is not sticky: a later activation retries and persists.
		const retry = await skillActivationStore.recordActivation({
			conversationId: 'conv-fail-retry',
			result: baseResult()
		});

		expect(retry.created).toBe(true);
		// The concurrent second call did not attempt its own persistence.
		expect(mockCreateMessageBranchPair).toHaveBeenCalledTimes(2);
		expect(skillActivationStore.isActivated('conv-fail-retry', 'opaque-id-1')).toBe(true);
	});

	it('recordActivation anchors a model read to the persisted assistant tool call carrying the model call id', async () => {
		mockGetConversationMessages.mockResolvedValue([assistantToolCallMessage('conv-model')]);

		const record = await skillActivationStore.recordActivation({
			conversationId: 'conv-model',
			result: baseResult({
				content_xml: '<skill_content name="demo-skill">body</skill_content>'
			}),
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

describe('Catalog preview reads are inert (Task 5)', () => {
	function jsonResponse(body: unknown, status = 200): Response {
		return new Response(JSON.stringify(body), {
			headers: { 'content-type': 'application/json' },
			status
		});
	}

	it('a preview read through SkillsService never creates a message or an activation record', async () => {
		vi.stubGlobal('fetch', vi.fn().mockResolvedValue(jsonResponse(baseResult())));

		const result = await SkillsService.read({ name: 'demo-skill' }, '/srv/project');

		expect(result.kind).toBe('skill');
		// Preview is non-durable and does not activate the session.
		expect(skillActivationStore.isActivated('conv-preview', 'opaque-id-1')).toBe(false);
		expect(mockCreateMessageBranch).not.toHaveBeenCalled();
		expect(mockCreateMessageBranchPair).not.toHaveBeenCalled();
	});
});
