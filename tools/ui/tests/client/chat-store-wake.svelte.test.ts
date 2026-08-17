// Guards queued agentic wakes after Skills activation with real store singletons.

// Form-level activation wakes successful flows but not failed or unavailable ones.

import ChatFormTestWrapper from './components/ChatFormTestWrapper.svelte';
import { MessageRole } from '$lib/enums';
import { ChatService } from '$lib/services/chat.service';
import { DatabaseService } from '$lib/services/database.service';
import { dispatchSkillActivation } from '$lib/services/skill-command.service';
import { agenticStore } from '$lib/stores/agentic.svelte';
import { chatStore } from '$lib/stores/chat.svelte';
import { conversationsStore } from '$lib/stores/conversations.svelte';
import { skillActivationStore } from '$lib/stores/skill-activation.svelte';
import { skillsStore } from '$lib/stores/skills.svelte';
import { skillAvailabilityStore } from '$lib/stores/skill-availability.svelte';
import type { DatabaseConversation, DatabaseMessage } from '$lib/types/database';
import type { SkillBaseReadResult } from '$lib/types/skills';
import type { SkillCatalogEntry } from '$lib/types';
import type { SkillCatalogSlot } from '$lib/stores/skills.svelte';
import { classifyLeafResume } from '$lib/utils';
import { tick } from 'svelte';
import { afterEach, beforeEach, describe, expect, it, vi } from 'vitest';
import { userEvent } from 'vitest/browser';
import { render } from 'vitest-browser-svelte';

// In-memory message tree backing the mocked DatabaseService.
const db = vi.hoisted(() => {
	const messages: DatabaseMessage[] = [];

	return {
		messages,
		seed(message: DatabaseMessage): void {
			messages.push(message);
		},
		add(message: DatabaseMessage): void {
			messages.push(message);

			if (message.parent) {
				const parent = messages.find((m) => m.id === message.parent);

				if (parent) parent.children = [...parent.children, message.id];
			}
		},
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

vi.mock('$lib/services/database.service', () => ({
	DatabaseService: {
		createMessageBranch: vi.fn(async (message: Omit<DatabaseMessage, 'id'>, parentId: string | null) => {
			const created: DatabaseMessage = {
				...message,
				children: [],
				id: `db-msg-${db.messages.length + 1}`,
				parent: parentId
			};

			db.add(created);

			return created;
		}),
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
		updateCurrentNode: vi.fn(async () => undefined),
		updateConversation: vi.fn(async () => undefined),
		updateMessage: vi.fn(async () => undefined)
	}
}));

vi.mock('$lib/services/chat.service', () => ({
	ChatService: {
		sendMessage: vi.fn(
			(messages: unknown, options: Record<string, unknown> & { onComplete?: (content: string) => void }) => {
				return new Promise<void>((resolve) => {
					streams.items.push({
						messages: messages as DatabaseMessage[],
						options,
						finish: () => {
							resolve();
							options.onComplete?.('queued wake answer');
						}
					});
				});
			}
		),
		selectActiveStream: vi.fn(() => null)
	}
}));

vi.mock('$lib/services/skill-command.service', () => ({ dispatchSkillActivation: vi.fn() }));
vi.mock('$lib/stores/skills.svelte', () => ({ skillsStore: { slotFor: vi.fn() } }));

const mockSendMessage = vi.mocked(ChatService.sendMessage);

const CONV_ID = 'conv-wake-1';

function baseResult(
	skill: { id: string; name: string },
	contentXml: string
): SkillBaseReadResult {
	return {
		body_markdown: `# ${skill.name}\nbody`,
		content_xml: contentXml,
		diagnostics: [],
		kind: 'skill',
		resources: { paths: [], truncated: false },
		skill: {
			id: skill.id,
			metadata: { description: `The ${skill.name} skill`, name: skill.name },
			name: skill.name,
			provider: 'project',
			scope: 'project'
		},
		source: `---\nname: ${skill.name}\n---\n# Body`
	};
}

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

// Form-level wake helpers

function formSkill(name: string, description = `${name} description`): SkillCatalogEntry {
	return {
		catalog_xml: '<skill />',
		description,
		id: `opaque-${name}`,
		instruction: { bytes: 1, lines: 1, modified_at: null, tokens: 1, tokens_estimated: false },
		name,
		provider: 'agents',
		resources: { count: 0, truncated: false },
		scope: 'global'
	};
}

function readySlot(entries: SkillCatalogEntry[]): SkillCatalogSlot {
	return {
		catalog: { catalog_instruction_xml: '', diagnostics: [], skills: entries },
		cwd: undefined,
		generation: 1,
		status: 'ready'
	};
}

async function selectSkill(name: string) {
	const { container } = render(ChatFormTestWrapper);

	await tick();

	const textarea = container.querySelector('textarea');

	if (!(textarea instanceof HTMLTextAreaElement)) throw new Error('textarea not rendered');

	await userEvent.click(textarea);
	await userEvent.keyboard(`/skills ${name}`);
	await tick();
	// The sole candidate is pre-highlighted on open; Enter selects it.
	await userEvent.keyboard('{Enter}');
	await tick();
}

beforeEach(async () => {
	db.reset();
	streams.items.length = 0;
	vi.clearAllMocks();

	conversationsStore.activeConversation = makeConversation();
	conversationsStore.activeMessages = [];

	vi.spyOn(agenticStore, 'runAgenticFlow').mockResolvedValue({ handled: false });

	// Form-level wake setup
	skillAvailabilityStore.setEnabled('opaque-frontend-design', true);
	skillAvailabilityStore.setEnabled('opaque-disabled', true);
	skillAvailabilityStore.setEnabled('opaque-enabled-a', true);
	skillAvailabilityStore.setEnabled('opaque-enabled-b', true);
	vi.mocked(skillsStore.slotFor).mockReturnValue(readySlot([formSkill('frontend-design')]));
	vi.mocked(dispatchSkillActivation).mockReset();
	vi.mocked(dispatchSkillActivation).mockResolvedValue({ ok: true, created: true });
});

afterEach(() => {
	vi.restoreAllMocks();
	conversationsStore.activeConversation = null;
	conversationsStore.activeMessages = [];
});

describe('chatStore.runTurnFromLeaf in an active conversation', () => {
	it('queues a wake requested while a previous wake is streaming and anchors it once the flow clears', async () => {
		// First /skills activation in a fresh-ish conversation (pair under root).
		const firstActivation = await skillActivationStore.recordActivation({
			conversationId: CONV_ID,
			cwd: undefined,
			result: baseResult(
				{ id: 'skill-frontend', name: 'frontend-design' },
				'<skill_content name="frontend-design">design</skill_content>'
			)
		});

		expect(firstActivation.created).toBe(true);

		// First wake: real routing from the tool-result leaf (fresh turn).
		const wakeOne = chatStore.runTurnFromLeaf();

		// The first wake must reach the stream sink and anchor at its pair leaf.
		await vi.waitFor(() => expect(streams.items.length).toBe(1));
		expect(streams.items[0].messages.at(-1)).toMatchObject({
			id: expect.any(String),
			role: MessageRole.TOOL
		});

		// A second activation occurs while the first wake streams.
		const secondActivation = await skillActivationStore.recordActivation({
			conversationId: CONV_ID,
			cwd: undefined,
			result: baseResult(
				{ id: 'skill-pdf', name: 'pdf' },
				'<skill_content name="pdf">pdf guide</skill_content>'
			)
		});

		expect(secondActivation.created).toBe(true);

		const wakeTwo = chatStore.runTurnFromLeaf();
		const pdfLeaf = conversationsStore.activeMessages.at(-1);

		expect(pdfLeaf?.role).toBe(MessageRole.TOOL);
		// No second stream while the first flow is still in flight.
		await tick();
		expect(streams.items.length).toBe(1);

		// Completing the first stream releases the queued wake.
		streams.items[0].finish();
		await wakeOne;
		await wakeTwo;

		await vi.waitFor(() => expect(streams.items.length).toBe(2));

		const anchored = streams.items[1].messages.at(-1);

		expect(anchored).toMatchObject({ id: pdfLeaf?.id, role: MessageRole.TOOL });
		expect(streams.items[1].messages).toContainEqual(
			expect.objectContaining({ role: MessageRole.ASSISTANT })
		);
		// The wake streamed a real turn (the sink's onComplete payload).
		expect(mockSendMessage).toHaveBeenCalledTimes(2);
	});
});

// Pure routing contract for waking a turn after a command-only activation:
// assistant leaves continue through the existing continuation machinery,
// tool result and user leaves open a fresh turn, and an empty conversation
// is a no-op.

describe('classifyLeafResume', () => {
	it('continues assistant leaves through the continuation machinery', () => {
		expect(classifyLeafResume(MessageRole.ASSISTANT)).toBe('continue-assistant');
	});

	it('opens a fresh turn after a tool result leaf', () => {
		expect(classifyLeafResume(MessageRole.TOOL)).toBe('fresh-turn');
	});

	it('opens a fresh turn after a user leaf', () => {
		expect(classifyLeafResume(MessageRole.USER)).toBe('fresh-turn');
	});

	it('is a no-op on an empty conversation', () => {
		expect(classifyLeafResume(undefined)).toBe('no-op');
	});
});

// Form-level wake contract: selecting a skill dispatches the durable
// activation and a successful outcome wakes the agentic loop through
// chatStore.runTurnFromLeaf. Not-found and unavailable outcomes never wake.

describe('/skills <name> wake', () => {
	it('wakes the agentic loop after a successful activation', async () => {
		const runTurn = vi.spyOn(chatStore, 'runTurnFromLeaf').mockResolvedValue();

		await selectSkill('frontend-design');

		await vi.waitFor(() =>
			expect(vi.mocked(dispatchSkillActivation)).toHaveBeenCalledWith('frontend-design')
		);
		await vi.waitFor(() => expect(runTurn).toHaveBeenCalledTimes(1));
	});

	it('omits disabled picker entries and preserves enabled server order', async () => {
		vi.mocked(skillsStore.slotFor).mockReturnValue(
			readySlot([formSkill('disabled'), formSkill('enabled-b'), formSkill('enabled-a')])
		);
		skillAvailabilityStore.setEnabled('opaque-disabled', false);

		const { container } = render(ChatFormTestWrapper);
		await tick();

		const textarea = container.querySelector('textarea');

		if (!(textarea instanceof HTMLTextAreaElement)) throw new Error('textarea not rendered');

		await userEvent.click(textarea);
		await userEvent.keyboard('/skills ');
		await tick();

		const rowText = Array.from(document.querySelectorAll<HTMLElement>('[data-picker-index]')).map(
			(row) => row.textContent ?? ''
		);

		expect(rowText).toHaveLength(2);
		expect(rowText[0]).toContain('enabled-b');
		expect(rowText[1]).toContain('enabled-a');
		expect(rowText.join(' ')).not.toContain('disabled');
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
});
