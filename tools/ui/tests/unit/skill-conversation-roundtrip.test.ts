import { NEWLINE } from '$lib/constants';
import { AttachmentType, MessageRole, MessageType } from '$lib/enums';
import { skillActivationExtra, isSkillExtra } from '$lib/services/skills-activation.service';
import { SKILL_READ_TOOL } from '$lib/services/skills-adapters.service';
import type { DatabaseMessage, ExportedConversation } from '$lib/types/database';
import type { SkillBaseReadResult } from '$lib/types/skills';
import { filterByLeafNodeId } from '$lib/utils';
import { beforeAll, describe, expect, it } from 'vitest';

let conversationsStore: typeof import('$lib/stores/conversations.svelte').conversationsStore;
let ChatService: typeof import('$lib/services/chat.service').ChatService;

// node env unit project has no DOM, install a minimal localStorage backed by a
// Map before the store modules read it (same pattern as
// conversation-import.test.ts).
beforeAll(async () => {
	const store = new Map<string, string>();
	const polyfill: Storage = {
		clear: () => store.clear(),
		getItem: (k) => (store.has(k) ? store.get(k)! : null),
		key: (i) => Array.from(store.keys())[i] ?? null,
		get length() {
			return store.size;
		},
		removeItem: (k) => {
			store.delete(k);
		},
		setItem: (k, v) => {
			store.set(k, String(v));
		}
	};

	(globalThis as unknown as { localStorage: Storage }).localStorage = polyfill;

	({ conversationsStore } = await import('$lib/stores/conversations.svelte'));
	({ ChatService } = await import('$lib/services/chat.service'));
}, 30000);

function baseResult(): SkillBaseReadResult {
	return {
		kind: 'skill',
		skill: {
			id: 'opaque-demo',
			name: 'demo-skill',
			scope: 'project',
			provider: 'agents',
			metadata: { name: 'demo-skill', description: 'A demo skill', license: 'MIT' }
		},
		resources: { paths: [], truncated: false },
		source: '---\nname: demo-skill\ndescription: A demo skill\nlicense: MIT\n---\n# Body',
		body_markdown: '# Body',
		content_xml: '<skill_content name="demo-skill">body &amp; more</skill_content>',
		diagnostics: []
	};
}

/** One agentic turn: user -> synthetic assistant tool call -> paired tool result. */
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
				id: toolCallId,
				type: 'function',
				function: { name: SKILL_READ_TOOL, arguments: JSON.stringify({ name: 'demo-skill' }) }
			}
		]),
		type: MessageType.TEXT
	};
	const toolResult: DatabaseMessage = {
		children: [],
		content: '<skill_content name="demo-skill">body &amp; more</skill_content>',
		convId,
		extra: [skillActivationExtra(baseResult())],
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

function makeSession(): ExportedConversation {
	const convId = 'conv-roundtrip';
	const root: DatabaseMessage = {
		children: ['user-1'],
		content: '',
		convId,
		id: 'root-1',
		parent: null,
		role: MessageRole.USER,
		timestamp: 0,
		type: MessageType.ROOT
	};
	const user: DatabaseMessage = {
		children: ['assistant-1'],
		content: 'hello',
		convId,
		id: 'user-1',
		parent: 'root-1',
		role: MessageRole.USER,
		timestamp: 1,
		type: MessageType.TEXT
	};
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
			{ type: AttachmentType.MCP_RESOURCE, name: 'r', serverName: 'srv', uri: 'file:///r', content: 'c' }
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
		messages: [root, user, assistant, toolResult, plainAssistant, mcpToolResult]
	};
}

describe('conversation export/import round trip with Skills metadata', () => {
	it('retains the typed SKILL metadata, the paired message structure, and ordinary non-Skills messages', async () => {
		const jsonl = conversationsStore.serializeSessionToJsonl(makeSession());
		const sessions: ExportedConversation[] = await conversationsStore.parseImportFile(
			new File([jsonl], 'export.jsonl')
		);

		expect(sessions).toHaveLength(1);

		const messages = sessions[0].messages;
		const assistant = messages.find((m) => m.id === 'assistant-1');
		const toolResult = messages.find((m) => m.id === 'tool-result-1');
		const plainAssistant = messages.find((m) => m.id === 'assistant-2');
		const mcpToolResult = messages.find((m) => m.id === 'tool-result-2');

		expect(assistant).toBeDefined();
		expect(toolResult).toBeDefined();
		expect(plainAssistant?.content).toBe('Plain non-Skills reply');
		expect(mcpToolResult?.extra).toEqual([
			{ type: AttachmentType.MCP_RESOURCE, name: 'r', serverName: 'srv', uri: 'file:///r', content: 'c' }
		]);

		// The synthetic assistant tool call still pairs with its tool result.
		const calls = JSON.parse(assistant!.toolCalls ?? '') as Array<{ id: string; function: { name: string } }>;

		expect(calls[0].function.name).toBe(SKILL_READ_TOOL);
		expect(toolResult!.toolCallId).toBe(calls[0].id);
		expect(toolResult!.content).toBe('<skill_content name="demo-skill">body &amp; more</skill_content>');

		// The typed durable metadata survives the round trip intact.
		const [extra] = toolResult!.extra ?? [];

		expect(isSkillExtra(extra)).toBe(true);
		expect(extra).toMatchObject({
			kind: 'base',
			skillId: 'opaque-demo',
			name: 'demo-skill',
			scope: 'project',
			provider: 'agents'
		});
		if (isSkillExtra(extra)) {
			expect(extra.metadata?.license).toBe('MIT');
		}
	});

	it('preserves the paired structure when the export is a multi-session JSONL file', async () => {
		const jsonl = [makeSession(), makeSession()]
			.map((s) => conversationsStore.serializeSessionToJsonl(s))
			.join(NEWLINE);
		const sessions: ExportedConversation[] = await conversationsStore.parseImportFile(
			new File([jsonl], 'multi.jsonl')
		);

		for (const session of sessions) {
			const assistant = session.messages.find((m) => m.id === 'assistant-1');
			const toolResult = session.messages.find((m) => m.id === 'tool-result-1');
			const calls = JSON.parse(assistant!.toolCalls ?? '') as Array<{ id: string }>;

			expect(toolResult!.toolCallId).toBe(calls[0].id);
			expect(isSkillExtra(toolResult!.extra?.[0])).toBe(true);
		}
	});

	it('falls back gracefully on malformed historical SKILL-shaped records', async () => {
		const session = makeSession();
		const toolResult = session.messages.find((m) => m.id === 'tool-result-1')!;

		toolResult.extra = [
			{ type: AttachmentType.SKILL, kind: 'base', state: 'approved', name: 'demo-skill', skillId: undefined }
		] as unknown as typeof toolResult.extra;

		const jsonl = conversationsStore.serializeSessionToJsonl(session);
		const sessions: ExportedConversation[] = await conversationsStore.parseImportFile(
			new File([jsonl], 'export.jsonl')
		);
		const [extra] = sessions[0].messages.find((m) => m.id === 'tool-result-1')!.extra ?? [];

		// The record survives as opaque data but is not a valid activation:
		// rendering falls back to the generic tool card.
		expect(isSkillExtra(extra)).toBe(false);
	});
});

describe('paired message validity for the model', () => {
	it('converts the synthetic pair to valid assistant tool_calls + tool result API messages', async () => {
		const [assistant, toolResult] = skillTurn('conv-roundtrip');
		const assistantApi = await ChatService.convertDbMessageToApiChatMessageData(assistant);
		const toolApi = await ChatService.convertDbMessageToApiChatMessageData(toolResult);

		const toolCalls = assistantApi.tool_calls ?? [];

		expect(assistantApi.role).toBe(MessageRole.ASSISTANT);
		expect(assistantApi.content).toBe('');
		expect(toolCalls[0].id).toBe('call_skill_1');
		expect(toolCalls[0].function!.name).toBe(SKILL_READ_TOOL);

		expect(toolApi.role).toBe(MessageRole.TOOL);
		expect(toolApi.tool_call_id).toBe('call_skill_1');
		expect(toolApi.content).toBe('<skill_content name="demo-skill">body &amp; more</skill_content>');
	});
});

describe('visible-path branch filtering preserves Skills metadata', () => {
	it('keeps the SKILL extra on the filtered root-to-leaf path and drops the off-path branch', () => {
		const convId = 'conv-branch';
		const root: DatabaseMessage = {
			children: ['user-1'],
			content: '',
			convId,
			id: 'root-1',
			parent: null,
			role: MessageRole.USER,
			timestamp: 0,
			type: MessageType.ROOT
		};
		const user1: DatabaseMessage = {
			children: ['assistant-1'],
			content: 'which skill?',
			convId,
			id: 'user-1',
			parent: 'root-1',
			role: MessageRole.USER,
			timestamp: 1,
			type: MessageType.TEXT
		};
		const assistant1: DatabaseMessage = {
			children: ['tool-result-1', 'user-2'],
			content: '',
			convId,
			id: 'assistant-1',
			parent: 'user-1',
			role: MessageRole.ASSISTANT,
			timestamp: 2,
			toolCalls: JSON.stringify([
				{
					id: 'call_1',
					type: 'function',
					function: { name: SKILL_READ_TOOL, arguments: '{"name":"demo-skill"}' }
				}
			]),
			type: MessageType.TEXT
		};
		const toolResult1: DatabaseMessage = {
			children: [],
			content: '<skill_content name="demo-skill">body</skill_content>',
			convId,
			extra: [skillActivationExtra(baseResult())],
			id: 'tool-result-1',
			parent: 'assistant-1',
			role: MessageRole.TOOL,
			timestamp: 3,
			toolCallId: 'call_1',
			toolCalls: '',
			type: MessageType.TEXT
		};
		const user2: DatabaseMessage = {
			children: ['assistant-2'],
			content: 'edited follow-up',
			convId,
			id: 'user-2',
			parent: 'assistant-1',
			role: MessageRole.USER,
			timestamp: 4,
			type: MessageType.TEXT
		};
		const assistant2: DatabaseMessage = {
			children: ['tool-result-2'],
			content: '',
			convId,
			id: 'assistant-2',
			parent: 'user-2',
			role: MessageRole.ASSISTANT,
			timestamp: 5,
			toolCalls: JSON.stringify([
				{
					id: 'call_2',
					type: 'function',
					function: { name: SKILL_READ_TOOL, arguments: '{"name":"demo-skill"}' }
				}
			]),
			type: MessageType.TEXT
		};
		const toolResult2: DatabaseMessage = {
			children: [],
			content: '<skill_content name="demo-skill">body</skill_content>',
			convId,
			extra: [skillActivationExtra(baseResult())],
			id: 'tool-result-2',
			parent: 'assistant-2',
			role: MessageRole.TOOL,
			timestamp: 6,
			toolCallId: 'call_2',
			toolCalls: '',
			type: MessageType.TEXT
		};
		const all = [root, user1, assistant1, toolResult1, user2, assistant2, toolResult2];

		// The visible-path compaction to branch B's leaf keeps the paired
		// SKILL tool result on the path and filters out branch A entirely.
		const pathB = filterByLeafNodeId(all, 'tool-result-2', false);

		expect(pathB.map((message) => message.id)).toEqual([
			'user-1',
			'assistant-1',
			'user-2',
			'assistant-2',
			'tool-result-2'
		]);
		// Same object reference: the filter copies nothing and strips nothing,
		// so the typed SKILL metadata survives intact.
		expect(pathB).toContain(toolResult2);
		expect(pathB).not.toContain(toolResult1);

		const [extra] = toolResult2.extra ?? [];

		expect(isSkillExtra(extra)).toBe(true);
		expect(extra).toMatchObject({
			kind: 'base',
			skillId: 'opaque-demo',
			name: 'demo-skill',
			scope: 'project',
			provider: 'agents'
		});

		// Navigating to branch A's leaf keeps ITS SKILL record and drops branch B.
		const pathA = filterByLeafNodeId(all, 'tool-result-1', false);

		expect(pathA.map((message) => message.id)).toEqual(['user-1', 'assistant-1', 'tool-result-1']);
		expect(pathA).toContain(toolResult1);
		expect(pathA).not.toContain(toolResult2);
	});
});
