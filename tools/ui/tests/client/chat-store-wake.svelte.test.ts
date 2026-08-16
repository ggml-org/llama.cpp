// Store-level wake contract: runTurnFromLeaf anchors a real agentic turn at the
// current leaf after a /skills activation, even when the activation lands while
// the previous wake's flow is still streaming. Pre-fix the loading guard
// silently dropped the second wake (pair persisted, no stream, no error); the
// wake must instead be queued and run once the in-flight flow clears.
//
// Uses the REAL chatStore/conversationsStore/skillActivationStore singletons
// with a mocked DatabaseService (in-memory message tree) and a controllable
// ChatService.sendMessage stream sink. agenticStore.runAgenticFlow is stubbed
// to fall through to the sink like a non-agentic stream.

import { MessageRole, MessageType } from '$lib/enums';
import { ChatService } from '$lib/services/chat.service';
import { DatabaseService } from '$lib/services/database.service';
import { agenticStore } from '$lib/stores/agentic.svelte';
import { chatStore } from '$lib/stores/chat.svelte';
import { conversationsStore } from '$lib/stores/conversations.svelte';
import { skillActivationStore } from '$lib/stores/skill-activation.svelte';
import type { DatabaseConversation, DatabaseMessage } from '$lib/types/database';
import type { SkillBaseReadResult } from '$lib/types/skills';
import { tick } from 'svelte';
import { afterEach, beforeEach, describe, expect, it, vi } from 'vitest';

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

// Controllable stream sink: every ChatService.sendMessage call parks its
// options + payload until the test finishes the stream explicitly.
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

beforeEach(async () => {
	db.reset();
	streams.items.length = 0;
	vi.clearAllMocks();

	conversationsStore.activeConversation = makeConversation();
	conversationsStore.activeMessages = [];

	vi.spyOn(agenticStore, 'runAgenticFlow').mockResolvedValue({ handled: false });

	// The store's internal loading state is per-conversation; the singletons
	// start clean within this file, so no explicit reset is required here.
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

		// While the first wake is still streaming, the user activates a second
		// skill exactly like ChatForm does: pair persisted + mirrored, then
		// runTurnFromLeaf.
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

		// The first wake's stream completes: loading clears and the queued
		// wake must fire, anchoring a fresh turn at the pdf pair's leaf.
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
