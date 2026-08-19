// Guards /skills activation, wake, form token handling, and the export/import round trip.

import ChatFormTestWrapper from './components/ChatFormTestWrapper.svelte';
import { baseResult, makeEntry, resourceResult } from '../fixtures/skills';
import { NEWLINE, SKILL_READ_TOOL } from '$lib/constants';
import { AttachmentType, MessageRole, MessageType } from '$lib/enums';
import { ChatService } from '$lib/services/chat.service';
import { DatabaseService } from '$lib/services/database.service';
import { dispatchSkillActivation } from '$lib/services/skill-command.service';
import { isSkillExtra, skillActivationExtra } from '$lib/services/skills-activation.service';
import { SkillsService } from '$lib/services/skills.service';
import { agenticStore } from '$lib/stores/agentic.svelte';
import { chatStore } from '$lib/stores/chat.svelte';
import { conversationsStore } from '$lib/stores/conversations.svelte';
import { skillActivationStore } from '$lib/stores/skill-activation.svelte';
import { skillAvailabilityStore } from '$lib/stores/skill-availability.svelte';
import type { SkillCatalogSlot } from '$lib/stores/skills.svelte';
import { skillsStore } from '$lib/stores/skills.svelte';
import type { SkillCatalogEntry } from '$lib/types';
import type { DatabaseConversation, DatabaseMessage, ExportedConversation } from '$lib/types/database';
import type { SkillBaseReadResult } from '$lib/types/skills';
import { tick } from 'svelte';
import { afterEach, beforeAll, beforeEach, describe, expect, it, vi } from 'vitest';
import { userEvent } from 'vitest/browser';
import { render } from 'vitest-browser-svelte';

// In-memory message tree backing the mocked DatabaseService.
const db = vi.hoisted(() => {
	const messages: DatabaseMessage[] = [];

	return {
		add(message: DatabaseMessage): void {
			messages.push(message);

			if (message.parent) {
				const parent = messages.find((m) => m.id === message.parent);

				if (parent) parent.children = [...parent.children, message.id];
			}
		},
		messages,
		reset(): void {
			messages.length = 0;
		}
	};
});
// Controllable stream sink for explicit test completion.
const streams = vi.hoisted(() => {
	const items: Array<{
		messages: DatabaseMessage[];
		options: Record<string, unknown>;
		finish: () => void;
	}> = [];

	return { items };
});

vi.mock('$app/navigation', async (importOriginal) => {
	const actual = await importOriginal<typeof import('$app/navigation')>();

	return { ...actual, goto: vi.fn() };
});

vi.mock('$lib/services/chat.service', async (importOriginal) => {
	const actual = await importOriginal<typeof import('$lib/services/chat.service')>();

	return {
		...actual,
		selectActiveStream: vi.fn(() => null),
		sendMessage: vi.fn(
			(
				messages: unknown,
				options: Record<string, unknown> & { onComplete?: (content: string) => void }
			) =>
				new Promise<void>((resolve) => {
					streams.items.push({
						finish: () => {
							resolve();
							options.onComplete?.('queued wake answer');
						},
						messages: messages as DatabaseMessage[],
						options
					});
				})
		)
	};
});

vi.mock('$lib/services/database.service', () => ({
	DatabaseService: {
		createConversation: vi.fn(async (name: string) => ({
			currNode: null,
			id: 'db-conv-1',
			lastModified: Date.now(),
			name
		})),
		createMessageBranch: vi.fn(
			async (message: Omit<DatabaseMessage, 'id'>, parentId: string | null) => {
				const created: DatabaseMessage = {
					...message,
					children: [],
					id: `db-msg-${db.messages.length + 1}`,
					parent: parentId
				};

				db.add(created);

				return created;
			}
		),
		createMessageBranchPair: vi.fn(
			async (
				assistant: Omit<DatabaseMessage, 'id'>,
				toolResult: Omit<DatabaseMessage, 'id'>,
				parentId: string | null
			) => {
				const pairAssistant: DatabaseMessage = {
					...assistant,
					children: [],
					id: `db-pair-a-${db.messages.length + 1}`,
					parent: parentId
				};
				const pairTool: DatabaseMessage = {
					...toolResult,
					children: [],
					id: `db-pair-t-${db.messages.length + 1}`,
					parent: pairAssistant.id
				};

				db.add(pairAssistant);
				db.add(pairTool);
				pairAssistant.children = [pairTool.id];

				return [pairAssistant, pairTool];
			}
		),
		createRootMessage: vi.fn(async () => 'db-root'),
		getConversationMessages: vi.fn(async (convId: string) =>
			db.messages
				.filter((message) => message.convId === convId)
				.sort((a, b) => a.timestamp - b.timestamp)
		),
		updateConversation: vi.fn(async () => undefined),
		updateCurrentNode: vi.fn(async () => undefined),
		updateMessage: vi.fn(async () => undefined)
	}
}));

vi.mock('$lib/services/skill-command.service', () => ({ dispatchSkillActivation: vi.fn() }));
vi.mock('$lib/services/skills.service', async (importOriginal) => {
	const actual = await importOriginal();

	return { ...actual, SkillsService: { read: vi.fn() } };
});
vi.mock('$lib/stores/skill-availability.svelte', () => ({
	skillAvailabilityStore: { isDisabled: vi.fn(() => false) }
}));
vi.mock('$lib/stores/skills.svelte', () => ({
	skillsStore: {
		ensureCatalog: vi.fn(async () => undefined),
		invalidate: vi.fn(),
		refresh: vi.fn(async () => undefined),
		slotFor: vi.fn()
	}
}));

const mockSendMessage = vi.mocked(ChatService.sendMessage);
const CONV_ID = 'conv-wake-1';

function makeConversation(): DatabaseConversation {
	return {
		currNode: null,
		id: CONV_ID,
		lastModified: Date.now(),
		mcpServerOverrides: [],
		name: 'Wake test',
		reasoningEffort: undefined,
		thinkingEnabled: false
	};
}

function wakeResult(id: string, name: string, contentXml: string): SkillBaseReadResult {
	return baseResult(name, {
		content_xml: contentXml,
		skill: {
			id,
			metadata: { description: `The ${name} skill`, name },
			name,
			provider: 'project',
			scope: 'project'
		}
	});
}

function readySlot(entries: SkillCatalogEntry[]): SkillCatalogSlot {
	return {
		catalog: { catalog_instruction_xml: '', diagnostics: [], skills: entries },
		cwd: undefined,
		generation: 1,
		status: 'ready'
	};
}

async function textareaOf(): Promise<HTMLTextAreaElement> {
	const { container } = render(ChatFormTestWrapper);

	await tick();

	const textarea = container.querySelector('textarea');

	if (!(textarea instanceof HTMLTextAreaElement)) throw new Error('textarea not rendered');

	return textarea;
}

async function selectSkill(name: string) {
	const textarea = await textareaOf();

	await userEvent.click(textarea);
	await userEvent.keyboard(`/skills ${name}`);
	await tick();
	// The sole candidate is pre-highlighted on open; Enter selects it.
	await userEvent.keyboard('{Enter}');
	await tick();
}

function pickerRows(): HTMLElement[] {
	return Array.from(document.querySelectorAll<HTMLElement>('[data-picker-index]'));
}

beforeEach(() => {
	db.reset();
	streams.items.length = 0;
	vi.clearAllMocks();

	conversationsStore.activeConversation = null;
	conversationsStore.activeMessages = [];
	conversationsStore.pendingCwd = null;

	vi.mocked(SkillsService.read).mockReset();
	vi.mocked(SkillsService.read).mockResolvedValue(baseResult('frontend-design'));
	vi.mocked(skillAvailabilityStore.isDisabled).mockReset();
	vi.mocked(skillAvailabilityStore.isDisabled).mockReturnValue(false);
	vi.mocked(dispatchSkillActivation).mockReset();
	vi.mocked(dispatchSkillActivation).mockResolvedValue({ created: true, ok: true });
	vi.mocked(skillsStore.slotFor).mockReset();
	vi.mocked(skillsStore.slotFor).mockReturnValue(readySlot([makeEntry('frontend-design')]));
	vi.spyOn(agenticStore, 'runAgenticFlow').mockResolvedValue({ handled: false });
});

afterEach(() => {
	vi.restoreAllMocks();
	conversationsStore.activeConversation = null;
	conversationsStore.activeMessages = [];
});

describe('dispatchSkillActivation', () => {
	let realDispatch: typeof dispatchSkillActivation;

	beforeAll(async () => {
		({ dispatchSkillActivation: realDispatch } = await vi.importActual<typeof import('$lib/services/skill-command.service')>(
			'$lib/services/skill-command.service'
		));
	});

	it('creates a Skill-named conversation and persists the activation in fresh state', async () => {
		const create = vi.spyOn(conversationsStore, 'createConversation');
		const record = vi.spyOn(skillActivationStore, 'recordActivation');

		const outcome = await realDispatch('frontend-design');

		expect(create).toHaveBeenCalledWith('Skill: frontend-design');
		expect(record).toHaveBeenCalledWith(expect.objectContaining({ conversationId: 'db-conv-1' }));
		expect(conversationsStore.activeConversation?.name).toBe('Skill: frontend-design');
		expect(skillActivationStore.isActivated('db-conv-1', 'opaque-frontend-design')).toBe(true);
		expect(outcome).toEqual({ created: true, ok: true });
	});

	it('reuses the active conversation and threads the pending CWD through read and record', async () => {
		conversationsStore.activeConversation = { ...makeConversation(), id: 'conv-active' };
		conversationsStore.pendingCwd = '/pending';
		const record = vi.spyOn(skillActivationStore, 'recordActivation');

		const outcome = await realDispatch('frontend-design');

		expect(vi.mocked(DatabaseService.createConversation)).not.toHaveBeenCalled();
		expect(SkillsService.read).toHaveBeenCalledWith(
			{ name: 'frontend-design' },
			'/pending',
			undefined
		);
		expect(record).toHaveBeenCalledWith(
			expect.objectContaining({ conversationId: 'conv-active', cwd: '/pending' })
		);
		expect(outcome).toEqual({ created: true, ok: true });
	});

	it.each(['unavailable', 'not-found', 'disabled', 'persistence-failed'] as const)(
		'reports %s without persisting an activation',
		async (reason) => {
			if (reason === 'unavailable') {
				vi.mocked(SkillsService.read).mockRejectedValue(new Error('boom'));
			} else if (reason === 'not-found') {
				vi.mocked(SkillsService.read).mockResolvedValue(resourceResult('frontend-design'));
			} else if (reason === 'disabled') {
				vi.mocked(SkillsService.read).mockResolvedValue(baseResult('frontend-design'));
				vi.mocked(skillAvailabilityStore.isDisabled).mockImplementation(
					(id) => id === 'opaque-frontend-design'
				);
			} else {
				vi.mocked(SkillsService.read).mockResolvedValue(baseResult('frontend-design'));
				vi.mocked(DatabaseService.createMessageBranchPair).mockRejectedValueOnce(
					new Error('db full')
				);
			}

			const outcome = await realDispatch('frontend-design');

			expect(outcome).toEqual({ ok: false, reason });

			if (reason === 'persistence-failed') {
				// The created conversation stays visible and deletable.
				expect(conversationsStore.activeConversation?.name).toBe('Skill: frontend-design');
			} else {
				expect(conversationsStore.activeConversation).toBeNull();
				expect(vi.mocked(DatabaseService.createMessageBranchPair)).not.toHaveBeenCalled();
			}
		}
	);
});

describe('/skills wake and form integration', () => {
	beforeEach(() => {
		conversationsStore.activeConversation = makeConversation();
	});

	it('wakes the agentic loop after a successful activation', async () => {
		const runTurn = vi.spyOn(chatStore, 'runTurnFromLeaf').mockResolvedValue();

		await selectSkill('frontend-design');

		await vi.waitFor(() =>
			expect(vi.mocked(dispatchSkillActivation)).toHaveBeenCalledWith('frontend-design')
		);
		await vi.waitFor(() => expect(runTurn).toHaveBeenCalledTimes(1));
	});

	it.each(['disabled', 'not-found', 'unavailable', 'persistence-failed'] as const)(
		'does not wake when the activation %s',
		async (reason) => {
			const runTurn = vi.spyOn(chatStore, 'runTurnFromLeaf').mockResolvedValue();

			vi.mocked(dispatchSkillActivation).mockResolvedValue({ ok: false, reason });

			await selectSkill('frontend-design');

			await vi.waitFor(() => expect(vi.mocked(dispatchSkillActivation)).toHaveBeenCalled());
			expect(runTurn).not.toHaveBeenCalled();
		}
	);


	it('auto-opens the picker with the trimmed query and dispatches exactly once on explicit selection, clearing the buffer', async () => {
		const textarea = await textareaOf();

		await userEvent.click(textarea);
		await userEvent.keyboard('/skills   frontend-design');
		await tick();

		// Typing opens the picker with the trimmed args; it never dispatches mid-typing.
		await vi.waitFor(() => expect(pickerRows()).toHaveLength(1));
		expect(pickerRows()[0].textContent).toContain('frontend-design');
		expect(textarea.value).toBe('/skills   frontend-design');
		expect(vi.mocked(dispatchSkillActivation)).not.toHaveBeenCalled();

		await userEvent.keyboard('{Enter}');
		await tick();

		expect(textarea.value).toBe('');
		expect(vi.mocked(dispatchSkillActivation)).toHaveBeenCalledTimes(1);
		expect(vi.mocked(dispatchSkillActivation)).toHaveBeenCalledWith('frontend-design');
		await vi.waitFor(() => expect(pickerRows()).toHaveLength(0));
	});

	it('keeps the retained token literal after Escape until the token changes', async () => {
		const textarea = await textareaOf();

		await userEvent.click(textarea);
		await userEvent.keyboard('/skills frontend-design');
		await tick();

		await vi.waitFor(() => expect(pickerRows()).toHaveLength(1));
		expect(vi.mocked(dispatchSkillActivation)).not.toHaveBeenCalled();

		// Escape closes the picker but leaves the token literal in the buffer.
		await userEvent.keyboard('{Escape}');
		await vi.waitFor(() => expect(pickerRows()).toHaveLength(0));

		expect(textarea.value).toBe('/skills frontend-design');
		expect(vi.mocked(dispatchSkillActivation)).not.toHaveBeenCalled();

		// Editing the token reopens the picker against the new query.
		await userEvent.keyboard('{Backspace}');
		await vi.waitFor(() => expect(pickerRows()).toHaveLength(1));

		expect(textarea.value).toBe('/skills frontend-desig');
		expect(vi.mocked(dispatchSkillActivation)).not.toHaveBeenCalled();
	});
});

describe('conversation export/import round trip with Skills metadata', () => {
	function demoResult(): SkillBaseReadResult {
		return baseResult('demo-skill', {
			content_xml: '<skill_content name="demo-skill">body &amp; more</skill_content>',
			skill: {
				id: 'opaque-demo',
				metadata: { description: 'A demo skill', license: 'MIT', name: 'demo-skill' },
				name: 'demo-skill',
				provider: 'agents',
				scope: 'project'
			}
		});
	}

	/** One agentic turn: synthetic assistant tool call paired with its tool result. */
	function skillTurn(convId: string): DatabaseMessage[] {
		const toolCallId = 'call_skill_1';
		const assistant: DatabaseMessage = {
			children: ['tool-result-1'],
			content: '',
			convId,
			id: 'assistant-1',
			parent: 'user-1',
			role: MessageRole.ASSISTANT,
			timestamp: 2,
			toolCalls: JSON.stringify([
				{
					function: { arguments: JSON.stringify({ name: 'demo-skill' }), name: SKILL_READ_TOOL },
					id: toolCallId,
					type: 'function'
				}
			]),
			type: MessageType.TEXT
		};
		const toolResult: DatabaseMessage = {
			children: [],
			content: '<skill_content name="demo-skill">body &amp; more</skill_content>',
			convId,
			extra: [skillActivationExtra(demoResult())],
			id: 'tool-result-1',
			parent: 'assistant-1',
			role: MessageRole.TOOL,
			timestamp: 3,
			toolCallId,
			toolCalls: '',
			type: MessageType.TEXT
		};

		return [assistant, toolResult];
	}

	/** A session holding the Skills pair plus ordinary non-Skills messages. */
	function makeSession(): ExportedConversation {
		const convId = 'conv-roundtrip';
		const [assistant, toolResult] = skillTurn(convId);
		const plainAssistant: DatabaseMessage = {
			children: [],
			content: 'Plain non-Skills reply',
			convId,
			id: 'assistant-2',
			parent: 'tool-result-1',
			role: MessageRole.ASSISTANT,
			timestamp: 4,
			toolCalls: '',
			type: MessageType.TEXT
		};
		const mcpToolResult: DatabaseMessage = {
			children: [],
			content: 'mcp output',
			convId,
			extra: [
				{
					content: 'c',
					name: 'r',
					serverName: 'srv',
					type: AttachmentType.MCP_RESOURCE,
					uri: 'file:///r'
				}
			],
			id: 'tool-result-2',
			parent: 'assistant-2',
			role: MessageRole.TOOL,
			timestamp: 5,
			toolCallId: 'call_mcp_1',
			toolCalls: '',
			type: MessageType.TEXT
		};

		return {
			conv: { currNode: 'tool-result-2', id: convId, lastModified: 0, name: 'Round trip' },
			messages: [assistant, toolResult, plainAssistant, mcpToolResult]
		};
	}

	it('round-trips the SKILL extra, the pairing, plain messages, and malformed records, and converts the pair for the model', async () => {
		const [session] = await conversationsStore.parseImportFile(
			new File([conversationsStore.serializeSessionToJsonl(makeSession())], 'export.jsonl')
		);
		const assistant = session.messages.find((m) => m.id === 'assistant-1');
		const toolResult = session.messages.find((m) => m.id === 'tool-result-1');
		const plainAssistant = session.messages.find((m) => m.id === 'assistant-2');
		const mcpToolResult = session.messages.find((m) => m.id === 'tool-result-2');

		expect(plainAssistant?.content).toBe('Plain non-Skills reply');
		expect(mcpToolResult?.extra).toEqual([
			{
				content: 'c',
				name: 'r',
				serverName: 'srv',
				type: AttachmentType.MCP_RESOURCE,
				uri: 'file:///r'
			}
		]);

		// The synthetic assistant tool call still pairs with its tool result.
		const calls = JSON.parse(assistant!.toolCalls ?? '') as Array<{
			id: string;
			function: { name: string };
		}>;

		expect(calls[0].function.name).toBe(SKILL_READ_TOOL);
		expect(toolResult!.toolCallId).toBe(calls[0].id);
		expect(toolResult!.content).toBe(
			'<skill_content name="demo-skill">body &amp; more</skill_content>'
		);

		// The typed durable metadata survives the round trip intact.
		const [extra] = toolResult!.extra ?? [];

		expect(isSkillExtra(extra)).toBe(true);
		expect(extra).toMatchObject({
			kind: 'base',
			name: 'demo-skill',
			provider: 'agents',
			scope: 'project',
			skillId: 'opaque-demo'
		});

		if (isSkillExtra(extra)) {
			expect(extra.metadata?.license).toBe('MIT');
		}

		// Malformed historical SKILL-shaped records fall back to the generic renderer.
		const malformed = makeSession();
		const malformedTool = malformed.messages.find((m) => m.id === 'tool-result-1')!;

		malformedTool.extra = [
			{
				kind: 'base',
				name: 'demo-skill',
				skillId: undefined,
				state: 'approved',
				type: AttachmentType.SKILL
			}
		] as unknown as typeof malformedTool.extra;

		const malformedSessions = await conversationsStore.parseImportFile(
			new File([conversationsStore.serializeSessionToJsonl(malformed)], 'export.jsonl')
		);
		const [malformedExtra] =
			malformedSessions[0].messages.find((m) => m.id === 'tool-result-1')!.extra ?? [];

		expect(isSkillExtra(malformedExtra)).toBe(false);

		// The persisted pair converts to valid assistant tool_calls + tool result API messages.
		const [assistantDb, toolResultDb] = skillTurn('conv-roundtrip');
		const assistantApi = await ChatService.convertDbMessageToApiChatMessageData(assistantDb);
		const toolApi = await ChatService.convertDbMessageToApiChatMessageData(toolResultDb);

		expect(assistantApi.role).toBe(MessageRole.ASSISTANT);
		expect(assistantApi.content).toBe('');
		expect(assistantApi.tool_calls?.[0].id).toBe('call_skill_1');
		expect(assistantApi.tool_calls?.[0].function?.name).toBe(SKILL_READ_TOOL);

		expect(toolApi.role).toBe(MessageRole.TOOL);
		expect(toolApi.tool_call_id).toBe('call_skill_1');
		expect(toolApi.content).toBe(
			'<skill_content name="demo-skill">body &amp; more</skill_content>'
		);
	});
});
