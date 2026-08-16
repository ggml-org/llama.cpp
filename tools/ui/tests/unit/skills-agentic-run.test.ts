import { SKILL_LIST_TOOL, SKILL_READ_TOOL } from '$lib/constants';
import { buildSkillRunSnapshot, serializeSkillCatalogEnvelope } from '$lib/services/skills-packing.service';
import { skillDenialResult } from '$lib/services/skills-adapters.service';
import { skillActivationExtra, skillResourceExtra } from '$lib/services/skills-activation.service';
import { SkillsService } from '$lib/services/skills.service';
import { ChatService } from '$lib/services';
import { agenticStore } from '$lib/stores/agentic.svelte';
import { skillsStore } from '$lib/stores/skills.svelte';
import { toolsStore } from '$lib/stores/tools.svelte';
import { settingsStore } from '$lib/stores/settings.svelte';
import { serverStore } from '$lib/stores/server.svelte';
import type {
	SkillBaseReadResult,
	SkillCatalogEntry,
	SkillCatalogResponse,
	SkillResourceReadResult
} from '$lib/types';
import type { AgenticFlowCallbacks } from '$lib/types/agentic';
import { MessageRole, ToolPermissionDecision } from '$lib/enums';
import { beforeEach, describe, expect, it, vi } from 'vitest';
import type { Mock } from 'vitest';

vi.mock('$lib/services/skills.service', () => ({
	SkillsService: { list: vi.fn(), read: vi.fn() }
}));
vi.mock('$lib/stores/skills.svelte', () => ({
	skillsStore: { createRunSnapshot: vi.fn() }
}));
const skillActivationMockState = vi.hoisted(() => ({
	store: {
		isActivated: vi.fn(() => false),
		loadConversation: vi.fn().mockResolvedValue(undefined),
		recordActivation: vi.fn()
	}
}));

vi.mock('$lib/stores/skill-activation.svelte', () => ({
	skillActivationStore: skillActivationMockState.store
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
	allTools: [] as {
		definition: { function: { name: string; parameters: Record<string, unknown> }; type: string };
		key: string;
	}[]
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
const mockGetEnabledToolsForLLM = vi.mocked(toolsStore.getEnabledToolsForLLM);
const mockRecordActivation = vi.mocked(skillActivationMockState.store.recordActivation);
const mockLoadConversation = vi.mocked(skillActivationMockState.store.loadConversation);

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
		source: `---\nname: ${name}\n---\n# Body`,
		body_markdown: '# Body',
		content_xml: `<skill_content name="${name}">body</skill_content>`,
		diagnostics: []
	};
}

function resourceResult(name: string, path: string): SkillResourceReadResult {
	return {
		kind: 'resource',
		skill: { id: `opaque-${name}`, name, scope: 'project', provider: 'agents' },
		resource: { path },
		content_xml: `<skill_resource name="${name}" path="${path}">data</skill_resource>`,
		diagnostics: []
	};
}

function dummyTool() {
	return {
		type: 'function' as const,
		function: { name: 'test_tool', parameters: { properties: {}, type: 'object' } }
	};
}

function makeCallbacks(): { callbacks: AgenticFlowCallbacks } & {
	createAssistantMessage: Mock;
	createToolResultMessage: Mock;
	onAssistantTurnComplete: Mock;
	onFlowComplete: Mock;
	onToolResultMessageCreated: Mock;
} {
	const createAssistantMessage = vi.fn().mockResolvedValue({ id: 'assistant-2' });
	const createToolResultMessage = vi.fn().mockResolvedValue({ id: 'tool-result-1' });
	const onAssistantTurnComplete = vi.fn().mockResolvedValue(undefined);
	const onFlowComplete = vi.fn();
	const onToolResultMessageCreated = vi.fn();

	return {
		callbacks: {
			createAssistantMessage,
			createToolResultMessage,
			onAssistantTurnComplete,
			onFlowComplete,
			onToolResultMessageCreated
		},
		createAssistantMessage,
		createToolResultMessage,
		onAssistantTurnComplete,
		onFlowComplete,
		onToolResultMessageCreated
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
	mockSettingsStore.config = { agenticMaxTurns: 100, maxSkillBudget: 2000 };
	mockGetEnabledToolsForLLM.mockReturnValue([dummyTool()]);
	toolsMockState.allTools = [{ definition: dummyTool(), key: 'builtin:test_tool' }];
	mockRecordActivation.mockResolvedValue({
		created: true,
		extra: skillActivationExtra(baseResult('demo-skill')),
		toolResultMessage: { id: 'recorded-tool-result' }
	});
	mockLoadConversation.mockResolvedValue(undefined);
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
		expect(mockLoadConversation).toHaveBeenCalledWith('conv-1');

		const tools = mockSendMessage.mock.calls[0][1].tools as { function: { name: string } }[];
		const names = tools.map((t) => t.function.name);

		expect(names).toEqual(['test_tool', SKILL_READ_TOOL]);

		// The run's own snapshot decorates the first-request messages with the
		// byte-preserved envelope (prepended system message).
		const firstMessages = mockSendMessage.mock.calls[0][0] as { role: string; content: string }[];

		expect(firstMessages[0].role).toBe(MessageRole.SYSTEM);
		expect(firstMessages[0].content).toBe(serializeSkillCatalogEnvelope(snapshot.catalog));
	});

	it('does not touch tools, snapshots, or the activation store when the budget is zero', async () => {
		mockSettingsStore.config = { agenticMaxTurns: 100, maxSkillBudget: 0 };
		mockSendMessage.mockResolvedValue(undefined);

		const result = await agenticStore.runAgenticFlow(runParams('conv-1', makeCallbacks().callbacks));

		expect(result).toEqual({ handled: true });
		expect(mockSnapshot).not.toHaveBeenCalled();
		expect(mockLoadConversation).not.toHaveBeenCalled();

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

	it('exposes both adapters for a partial envelope: read_skill and list_skill', async () => {
		mockSettingsStore.config = { agenticMaxTurns: 100, maxSkillBudget: 1 };
		mockSnapshot.mockResolvedValue(
			buildSkillRunSnapshot('/run-cwd', makeCatalog('demo-skill', 'other-skill'))
		);
		mockSendMessage.mockResolvedValue(undefined);

		const result = await agenticStore.runAgenticFlow(runParams('conv-1', makeCallbacks().callbacks));

		expect(result).toEqual({ handled: true });

		const tools = mockSendMessage.mock.calls[0][1].tools as { function: { name: string } }[];

		expect(tools.map((t) => t.function.name)).toEqual([
			'test_tool',
			SKILL_READ_TOOL,
			SKILL_LIST_TOOL
		]);
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

	it('denies an unapproved base read with a structured no-content tool result and no activation', async () => {
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
		expect(mockRecordActivation).not.toHaveBeenCalled();
	});

	it('routes an approved base read through the shared durable operation, persisting the store-created tool result once', async () => {
		const snapshot = buildSkillRunSnapshot('/run-cwd', makeCatalog('demo-skill'));

		mockSnapshot.mockResolvedValue(snapshot);
		mockRead.mockResolvedValue(baseResult('demo-skill'));
		mockToolCallTurn(readSkillToolCallJson());

		const { callbacks, createToolResultMessage, onToolResultMessageCreated } = makeCallbacks();
		const runPromise = agenticStore.runAgenticFlow(runParams('conv-1', callbacks));

		await waitForPermission('conv-1');
		agenticStore.resolvePermission('conv-1', ToolPermissionDecision.ONCE);

		const result = await runPromise;

		expect(result).toEqual({ handled: true });
		expect(mockRecordActivation).toHaveBeenCalledWith(
			expect.objectContaining({
				conversationId: 'conv-1',
				cwd: '/run-cwd',
				toolCallId: 'call_1'
			})
		);
		// The shared operation already created the paired tool result message:
		// the flow must not create a second one, and must update its own
		// parent pointer so the next turn anchors to the new leaf.
		expect(createToolResultMessage).not.toHaveBeenCalled();
		expect(onToolResultMessageCreated).toHaveBeenCalledWith('recorded-tool-result');
	});

	it('persists a resource read tool result through the flow with its typed metadata attached', async () => {
		const snapshot = buildSkillRunSnapshot('/run-cwd', makeCatalog('demo-skill'));

		mockSnapshot.mockResolvedValue(snapshot);
		mockRead.mockResolvedValue(resourceResult('demo-skill', 'refs/DETAILS.md'));
		mockRecordActivation.mockResolvedValue({
			created: false,
			extra: skillResourceExtra(resourceResult('demo-skill', 'refs/DETAILS.md')),
			toolResultMessage: null
		});
		mockToolCallTurn(
			JSON.stringify([
				{
					id: 'call_1',
					type: 'function',
					function: {
						name: SKILL_READ_TOOL,
						arguments: '{"name":"demo-skill","path":"refs/DETAILS.md"}'
					}
				}
			])
		);

		const { callbacks, createToolResultMessage } = makeCallbacks();
		const runPromise = agenticStore.runAgenticFlow(runParams('conv-1', callbacks));

		await waitForPermission('conv-1');
		agenticStore.resolvePermission('conv-1', ToolPermissionDecision.ONCE);

		const result = await runPromise;

		expect(result).toEqual({ handled: true });
		expect(mockRecordActivation).toHaveBeenCalled();
		expect(createToolResultMessage).toHaveBeenCalledWith(
			'call_1',
			'<skill_resource name="demo-skill" path="refs/DETAILS.md">data</skill_resource>',
			[expect.objectContaining({ kind: 'resource', path: 'refs/DETAILS.md' })]
		);
	});
});
