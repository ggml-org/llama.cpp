import { buildSkillRunSnapshot, serializeSkillCatalogEnvelope } from '$lib/services/skills-packing.service';
import { SKILL_READ_TOOL, skillDenialResult } from '$lib/services/skills-adapters.service';
import { SkillsService } from '$lib/services/skills.service';
import { ChatService } from '$lib/services';
import { agenticStore } from '$lib/stores/agentic.svelte';
import { skillsStore } from '$lib/stores/skills.svelte';
import { toolsStore } from '$lib/stores/tools.svelte';
import { settingsStore } from '$lib/stores/settings.svelte';
import { serverStore } from '$lib/stores/server.svelte';
import type { SkillBaseReadResult, SkillCatalogEntry, SkillCatalogResponse } from '$lib/types';
import type { AgenticFlowCallbacks } from '$lib/types/agentic';
import { MessageRole, ToolPermissionDecision } from '$lib/enums';
import { beforeEach, describe, expect, it, vi } from 'vitest';

vi.mock('$lib/services/skills.service', () => ({
	SkillsService: { list: vi.fn(), read: vi.fn() }
}));
vi.mock('$lib/stores/skills.svelte', () => ({
	skillsStore: { createRunSnapshot: vi.fn() }
}));
vi.mock('$lib/services/chat.service', () => ({
	ChatService: {
		convertDbMessageToApiChatMessageData: vi.fn(),
		sendMessage: vi.fn()
	}
}));
vi.mock('$lib/services/tools.service', () => ({
	ToolsService: { executeTool: vi.fn(), executeToolRaw: vi.fn(), streamTool: vi.fn() }
}));
vi.mock('$lib/services/sandbox.service', () => ({
	SandboxService: { executeTool: vi.fn() }
}));
const toolsMockState = vi.hoisted(() => ({
	allTools: [] as { definition: { function: { name: string; parameters: Record<string, unknown> } }; key: string }[]
}));

vi.mock('$lib/stores/tools.svelte', () => ({
	toolsStore: {
		get allTools() {
			return toolsMockState.allTools;
		},
		builtinTools: [
			{ function: { name: 'test_tool', parameters: { properties: {}, type: 'object' } }, type: 'function' }
		],
		customTools: [],
		fetchBuiltinTools: vi.fn(),
		frontendTools: [],
		getEnabledToolsForLLM: vi.fn(),
		getPermissionKey: vi.fn(() => null),
		getToolServerLabel: vi.fn(() => ''),
		getToolSource: vi.fn(() => null),
		loading: false
	}
}));
vi.mock('$lib/stores/mcp.svelte', () => ({
	mcpStore: {
		acquireConnection: vi.fn(),
		ensureInitialized: vi.fn(),
		executeTool: vi.fn().mockResolvedValue({ content: 'mcp-ok', isError: false }),
		hasEnabledServers: vi.fn(() => false),
		releaseConnection: vi.fn()
	}
}));
vi.mock('$lib/stores/models.svelte', () => ({
	modelsStore: {
		isModelLoaded: vi.fn(() => false),
		modelSupportsVision: vi.fn(() => false),
		models: []
	}
}));
vi.mock('$lib/stores/permissions.svelte', () => ({
	permissionsStore: { allowTool: vi.fn(), allowTools: vi.fn(), hasTool: vi.fn(() => false) }
}));
vi.mock('$lib/stores/conversations.svelte', () => ({
	conversationsStore: { activeConversation: { cwd: '/run-cwd' } }
}));
vi.mock('$lib/stores/server.svelte', () => ({
	serverStore: { isRouterMode: false }
}));
vi.mock('$lib/stores/settings.svelte', () => ({
	settingsStore: { config: { agenticMaxTurns: 100, maxSkillBudget: 2000 } }
}));

const mockSnapshot = vi.mocked(skillsStore.createRunSnapshot);
const mockSendMessage = vi.mocked(ChatService.sendMessage);
const mockRead = vi.mocked(SkillsService.read);
const mockSettingsStore = vi.mocked(settingsStore);
const mockServerStore = vi.mocked(serverStore);
const mockGetEnabledToolsForLLM = vi.mocked(toolsStore.getEnabledToolsForLLM);

function makeEntry(name: string): SkillCatalogEntry {
	return {
		id: `opaque-${name}`,
		name,
		description: `description of ${name}`,
		scope: 'project',
		provider: 'agents',
		instruction: { bytes: 16, lines: 1, tokens: 4, tokens_estimated: true, modified_at: null },
		resources: { count: 0, truncated: false },
		catalog_xml: `<skill><name>${name}</name></skill>`
	};
}

function makeCatalog(...names: string[]): SkillCatalogResponse {
	return {
		skills: names.map(makeEntry),
		catalog_instruction_xml: '<available_skills>Call read_skill(name) when matching.</available_skills>',
		diagnostics: []
	};
}

function baseResult(name: string): SkillBaseReadResult {
	return {
		kind: 'skill',
		skill: { id: `opaque-${name}`, name, scope: 'project', provider: 'agents' },
		resources: { paths: [], truncated: false },
		content_xml: `<skill_content name="${name}">body</skill_content>`,
		diagnostics: []
	};
}

function dummyTool() {
	return {
		type: 'function' as const,
		function: { name: 'test_tool', parameters: { properties: {}, type: 'object' } }
	};
}

function makeCallbacks(): { callbacks: AgenticFlowCallbacks } & Record<string, ReturnType<typeof vi.fn>> {
	const createAssistantMessage = vi.fn().mockResolvedValue({ id: 'assistant-2' });
	const createToolResultMessage = vi.fn().mockResolvedValue({ id: 'tool-result-1' });
	const onAssistantTurnComplete = vi.fn().mockResolvedValue(undefined);
	const onFlowComplete = vi.fn();

	return {
		callbacks: {
			createAssistantMessage,
			createToolResultMessage,
			onAssistantTurnComplete,
			onFlowComplete
		},
		createAssistantMessage,
		createToolResultMessage,
		onAssistantTurnComplete,
		onFlowComplete
	};
}

function runParams(
	convId: string,
	callbacks: AgenticFlowCallbacks,
	overrides: Record<string, unknown> = {}
): Parameters<typeof agenticStore.runAgenticFlow>[0] {
	return {
		callbacks,
		conversationId: convId,
		messages: [{ content: 'hi', role: MessageRole.USER }],
		perChatOverrides: [],
		...overrides
	} as Parameters<typeof agenticStore.runAgenticFlow>[0];
}

async function waitForPermission(convId: string) {
	const deadline = Date.now() + 3000;

	while (Date.now() < deadline) {
		const pending = agenticStore.pendingPermissionRequest(convId);

		if (pending) return pending;

		await new Promise((r) => setTimeout(r, 5));
	}

	throw new Error('timed out waiting for a pending permission request');
}

function mockToolCallTurn(toolCallJson: string): void {
	// Counter-based base implementation (no once-queues, so nothing leaks
	// between tests): the first LLM turn emits the tool call, later turns
	// resolve with no calls and end the loop.
	let callIndex = 0;

	mockSendMessage.mockImplementation(async (_messages, options) => {
		callIndex += 1;

		if (callIndex === 1) {
			options.onToolCallChunk?.(toolCallJson);
		}
	});
}

function readSkillToolCallJson(): string {
	return JSON.stringify([
		{
			id: 'call_1',
			type: 'function',
			function: { name: SKILL_READ_TOOL, arguments: '{"name":"demo-skill"}' }
		}
	]);
}

beforeEach(() => {
	vi.clearAllMocks();
	mockServerStore.isRouterMode = false;
	mockSettingsStore.config = { agenticMaxTurns: 100, maxSkillBudget: 2000 };
	mockGetEnabledToolsForLLM.mockReturnValue([dummyTool()]);
	toolsMockState.allTools = [{ definition: dummyTool(), key: 'builtin:test_tool' }];
	agenticStore.clearSession('conv-1');
});

describe('agenticStore.runAgenticFlow Skills integration', () => {
	it('creates exactly one immutable snapshot before the agentic gate and registers snapshot adapters', async () => {
		const snapshot = buildSkillRunSnapshot('/run-cwd', makeCatalog('demo-skill'));

		mockSnapshot.mockResolvedValue(snapshot);
		mockSendMessage.mockResolvedValue(undefined);

		const result = await agenticStore.runAgenticFlow(runParams('conv-1', makeCallbacks().callbacks));

		expect(result).toEqual({ handled: true });
		expect(mockSnapshot).toHaveBeenCalledTimes(1);
		expect(mockSnapshot).toHaveBeenCalledWith('/run-cwd', undefined);

		const tools = mockSendMessage.mock.calls[0][1].tools as { function: { name: string } }[];
		const names = tools.map((t) => t.function.name);

		expect(names).toEqual(['test_tool', SKILL_READ_TOOL]);

		// The run's own snapshot decorates the first-request messages with the
		// byte-preserved envelope (prepended system message).
		const firstMessages = mockSendMessage.mock.calls[0][0] as { role: string; content: string }[];

		expect(firstMessages[0].role).toBe(MessageRole.SYSTEM);
		expect(firstMessages[0].content).toBe(serializeSkillCatalogEnvelope(snapshot.catalog));
	});

	it('does not touch tools or fetch a snapshot when the budget is zero', async () => {
		mockSettingsStore.config = { agenticMaxTurns: 100, maxSkillBudget: 0 };
		mockSendMessage.mockResolvedValue(undefined);

		const result = await agenticStore.runAgenticFlow(runParams('conv-1', makeCallbacks().callbacks));

		expect(result).toEqual({ handled: true });
		expect(mockSnapshot).not.toHaveBeenCalled();

		const tools = mockSendMessage.mock.calls[0][1].tools as { function: { name: string } }[];

		expect(tools.map((t) => t.function.name)).toEqual(['test_tool']);
	});

	it('registers no adapters when the snapshot is unavailable, leaving the run unchanged', async () => {
		mockSnapshot.mockRejectedValue(new Error('skills disabled'));
		mockSendMessage.mockResolvedValue(undefined);

		const result = await agenticStore.runAgenticFlow(runParams('conv-1', makeCallbacks().callbacks));

		expect(result).toEqual({ handled: true });
		expect(mockSendMessage).toHaveBeenCalledTimes(1);

		const tools = mockSendMessage.mock.calls[0][1].tools as { function: { name: string } }[];

		expect(tools.map((t) => t.function.name)).toEqual(['test_tool']);
	});

	it('registers no adapters for an empty catalog', async () => {
		mockSnapshot.mockResolvedValue(buildSkillRunSnapshot('/run-cwd', makeCatalog()));
		mockSendMessage.mockResolvedValue(undefined);

		const result = await agenticStore.runAgenticFlow(runParams('conv-1', makeCallbacks().callbacks));

		expect(result).toEqual({ handled: true });
		expect(mockSnapshot).toHaveBeenCalledTimes(1);

		const tools = mockSendMessage.mock.calls[0][1].tools as { function: { name: string } }[];

		expect(tools.map((t) => t.function.name)).toEqual(['test_tool']);
	});

	it('omits a colliding Skills adapter in favor of the existing registry tool', async () => {
		const snapshot = buildSkillRunSnapshot('/run-cwd', makeCatalog('demo-skill'));

		mockSnapshot.mockResolvedValue(snapshot);
		mockSendMessage.mockResolvedValue(undefined);
		toolsMockState.allTools = [
			{
				definition: { function: { name: SKILL_READ_TOOL, parameters: {} }, type: 'function' },
				key: 'custom:read_skill'
			},
			{ definition: dummyTool(), key: 'builtin:test_tool' }
		];

		const result = await agenticStore.runAgenticFlow(runParams('conv-1', makeCallbacks().callbacks));

		expect(result).toEqual({ handled: true });

		const tools = mockSendMessage.mock.calls[0][1].tools as { function: { name: string } }[];

		expect(tools.map((t) => t.function.name)).toEqual(['test_tool']);
	});

	it('keeps the generic non-Skills permission path untouched (no skill metadata, abort-safe signal)', async () => {
		mockSnapshot.mockResolvedValue(buildSkillRunSnapshot('/run-cwd', makeCatalog('demo-skill')));
		mockToolCallTurn(
			JSON.stringify([
				{
					id: 'call_1',
					type: 'function',
					function: { name: 'test_tool', arguments: '{}' }
				}
			])
		);

		const { callbacks, createToolResultMessage } = makeCallbacks();
		const runPromise = agenticStore.runAgenticFlow(runParams('conv-1', callbacks));

		const pending = await waitForPermission('conv-1');

		expect(pending).toEqual({ serverLabel: '', toolName: 'test_tool' });
		expect('skill' in pending).toBe(false);

		agenticStore.resolvePermission('conv-1', ToolPermissionDecision.ONCE);

		const result = await runPromise;

		expect(result).toEqual({ handled: true });
		expect(createToolResultMessage).toHaveBeenCalledWith('call_1', 'mcp-ok', undefined);
	});

	it('denies an unapproved base read with a structured no-content tool result', async () => {
		const snapshot = buildSkillRunSnapshot('/run-cwd', makeCatalog('demo-skill'));

		mockSnapshot.mockResolvedValue(snapshot);
		mockRead.mockResolvedValue(baseResult('demo-skill'));
		mockToolCallTurn(readSkillToolCallJson());

		const { callbacks, createToolResultMessage } = makeCallbacks();
		const runPromise = agenticStore.runAgenticFlow(runParams('conv-1', callbacks));

		const pending = await waitForPermission('conv-1');

		expect(pending).toEqual({
			serverLabel: 'Skills',
			skill: { name: 'demo-skill', provider: 'agents', scope: 'project' },
			toolName: SKILL_READ_TOOL
		});

		agenticStore.resolvePermission('conv-1', ToolPermissionDecision.DENY);

		const result = await runPromise;

		expect(result).toEqual({ handled: true });
		expect(createToolResultMessage).toHaveBeenCalledWith('call_1', skillDenialResult(SKILL_READ_TOOL), undefined);
	});

	it('allows an approved base read and persists the byte-preserved XML content', async () => {
		const snapshot = buildSkillRunSnapshot('/run-cwd', makeCatalog('demo-skill'));

		mockSnapshot.mockResolvedValue(snapshot);
		mockRead.mockResolvedValue(baseResult('demo-skill'));
		mockToolCallTurn(readSkillToolCallJson());

		const { callbacks, createToolResultMessage } = makeCallbacks();
		const runPromise = agenticStore.runAgenticFlow(runParams('conv-1', callbacks));

		await waitForPermission('conv-1');
		agenticStore.resolvePermission('conv-1', ToolPermissionDecision.ONCE);

		const result = await runPromise;

		expect(result).toEqual({ handled: true });
		expect(createToolResultMessage).toHaveBeenCalledWith(
			'call_1',
			'<skill_content name="demo-skill">body</skill_content>',
			undefined
		);
	});
});
