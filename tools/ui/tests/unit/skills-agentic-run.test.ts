// Guards the agentic flow's Skills integration: snapshot creation, adapter
// registration, and consent routing through the shared durable operation.

import { SKILL_LIST_TOOL, SKILL_READ_TOOL } from '$lib/constants';
import { MessageRole, ToolPermissionDecision } from '$lib/enums';
import { ChatService } from '$lib/services';
import * as SkillsServiceModule from '$lib/services/skills.service';
import { SkillsService } from '$lib/services/skills.service';
import { buildSkillRunSnapshot, serializeSkillCatalogEnvelope } from '$lib/services/skills.service';
import { skillActivationExtra, skillResourceExtra } from '$lib/services/skills-activation.service';
import { skillDenialResult } from '$lib/services/skills-adapters.service';
import { agenticStore } from '$lib/stores/agentic.svelte';
import { settingsStore } from '$lib/stores/settings.svelte';
import { skillsStore } from '$lib/stores/skills.svelte';
import { toolsStore } from '$lib/stores/tools.svelte';
import type { AgenticFlowCallbacks } from '$lib/types/agentic';
import type { Mock } from 'vitest';
import { beforeEach, describe, expect, it, vi } from 'vitest';
import { baseResult, catalogOf, resourceResult } from '../fixtures/skills';

vi.mock('$lib/services/skills.service', async (importOriginal) => {
	const actual = await importOriginal<typeof SkillsServiceModule>();

	return {
		...actual,
		SkillsService: { list: vi.fn(), read: vi.fn() }
	};
});
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
		browserTools: [],
		customTools: [],
		fetchServerTools: vi.fn(),
		getEnabledSkillToolNames: vi.fn(),
		getEnabledToolsForLLM: vi.fn(),
		getPermissionKey: vi.fn(() => null),
		getToolServerLabel: vi.fn(() => ''),
		getToolSource: vi.fn(() => null),
		loading: false,
		serverTools: [
			{
				function: { name: 'test_tool', parameters: { properties: {}, type: 'object' } },
				type: 'function'
			}
		]
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
		models: [],
		modelSupportsVision: vi.fn(() => false)
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
const mockGetEnabledSkillToolNames = vi.mocked(toolsStore.getEnabledSkillToolNames);
const mockRecordActivation = vi.mocked(skillActivationMockState.store.recordActivation);
const mockLoadConversation = vi.mocked(skillActivationMockState.store.loadConversation);

function dummyTool() {
	return {
		function: { name: 'test_tool', parameters: { properties: {}, type: 'object' } },
		type: 'function' as const
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
	// The first mock turn emits a tool call; later turns end the loop.
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
			function: { arguments: '{"name":"demo-skill"}', name: SKILL_READ_TOOL },
			id: 'call_1',
			type: 'function'
		}
	]);
}

function resourceSkillToolCallJson(): string {
	return JSON.stringify([
		{
			function: {
				arguments: '{"name":"demo-skill","path":"refs/DETAILS.md"}',
				name: SKILL_READ_TOOL
			},
			id: 'call_1',
			type: 'function'
		}
	]);
}

function toolNamesOfFirstSend(): string[] {
	const tools = mockSendMessage.mock.calls[0][1].tools as { function: { name: string } }[];

	return tools.map((t) => t.function.name);
}

beforeEach(() => {
	vi.clearAllMocks();
	mockSettingsStore.config = { agenticMaxTurns: 100, maxSkillBudget: 2000 };
	mockGetEnabledToolsForLLM.mockReturnValue([dummyTool()]);
	mockGetEnabledSkillToolNames.mockReturnValue(new Set([SKILL_READ_TOOL, SKILL_LIST_TOOL]));
	toolsMockState.allTools = [{ definition: dummyTool(), key: 'server:test_tool' }];
	mockRecordActivation.mockResolvedValue({
		created: true,
		extra: skillActivationExtra(baseResult('demo-skill')),
		toolResultMessage: { id: 'recorded-tool-result' }
	});
	mockLoadConversation.mockResolvedValue(undefined);
	agenticStore.clearSession('conv-1');
});

describe('agenticStore.runAgenticFlow Skills integration', () => {
	it('creates one immutable snapshot before the agentic gate, registering snapshot adapters with the byte-preserved envelope', async () => {
		const snapshot = buildSkillRunSnapshot('/run-cwd', catalogOf('demo-skill'));

		mockSnapshot.mockResolvedValue(snapshot);
		mockSendMessage.mockResolvedValue(undefined);

		const result = await agenticStore.runAgenticFlow(
			runParams('conv-1', makeCallbacks().callbacks)
		);

		expect(result).toEqual({ handled: true });
		expect(mockSnapshot).toHaveBeenCalledTimes(1);
		expect(mockSnapshot).toHaveBeenCalledWith('/run-cwd', undefined);
		expect(mockLoadConversation).toHaveBeenCalledWith('conv-1');
		expect(toolNamesOfFirstSend()).toEqual(['test_tool', SKILL_READ_TOOL]);

		// The run snapshot prepends the byte-preserved envelope.
		const firstMessages = mockSendMessage.mock.calls[0][0] as { role: string; content: string }[];

		expect(firstMessages[0].role).toBe(MessageRole.SYSTEM);
		expect(firstMessages[0].content).toBe(serializeSkillCatalogEnvelope(snapshot.catalog));
	});

	it('exposes both adapters for a partial envelope and none for a zero budget, empty catalog, or failed snapshot', async () => {
		mockSettingsStore.config = { agenticMaxTurns: 100, maxSkillBudget: 1 };
		mockSnapshot.mockResolvedValue(
			buildSkillRunSnapshot('/run-cwd', catalogOf('demo-skill', 'other-skill'))
		);
		mockSendMessage.mockResolvedValue(undefined);

		await agenticStore.runAgenticFlow(runParams('conv-1', makeCallbacks().callbacks));

		expect(toolNamesOfFirstSend()).toEqual(['test_tool', SKILL_READ_TOOL, SKILL_LIST_TOOL]);

		// Zero budget: no snapshot call, no adapters.
		mockSettingsStore.config = { agenticMaxTurns: 100, maxSkillBudget: 0 };
		mockSendMessage.mockClear();
		mockSnapshot.mockClear();
		agenticStore.clearSession('conv-zero');
		await agenticStore.runAgenticFlow(runParams('conv-zero', makeCallbacks().callbacks));

		expect(mockSendMessage).toHaveBeenCalled();

		// Failed snapshot: the run proceeds unchanged.
		mockSnapshot.mockRejectedValue(new Error('skills disabled'));
		mockSendMessage.mockClear();

		await agenticStore.runAgenticFlow(runParams('conv-1', makeCallbacks().callbacks));

	});

	it('denies an unapproved base read with a structured no-content tool result and no activation', async () => {
		mockSnapshot.mockResolvedValue(buildSkillRunSnapshot('/run-cwd', catalogOf('demo-skill')));
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

		await runPromise;

		expect(createToolResultMessage).toHaveBeenCalledWith(
			'call_1',
			skillDenialResult(SKILL_READ_TOOL),
			undefined
		);
		expect(mockRecordActivation).not.toHaveBeenCalled();
	});

	it('routes an approved base read through the shared durable operation, persisting the store-created tool result once', async () => {
		mockSnapshot.mockResolvedValue(buildSkillRunSnapshot('/run-cwd', catalogOf('demo-skill')));
		mockRead.mockResolvedValue(baseResult('demo-skill'));
		mockToolCallTurn(readSkillToolCallJson());

		const { callbacks, createToolResultMessage, onToolResultMessageCreated } = makeCallbacks();
		const runPromise = agenticStore.runAgenticFlow(runParams('conv-1', callbacks));

		await waitForPermission('conv-1');
		agenticStore.resolvePermission('conv-1', ToolPermissionDecision.ONCE);

		await runPromise;

		expect(mockRecordActivation).toHaveBeenCalledWith(
			expect.objectContaining({
				conversationId: 'conv-1',
				cwd: '/run-cwd',
				toolCallId: 'call_1'
			})
		);
		// Reuse the shared tool result and advance the flow leaf.
		expect(createToolResultMessage).not.toHaveBeenCalled();
		expect(onToolResultMessageCreated).toHaveBeenCalledWith('recorded-tool-result');
	});

	it('persists a resource read tool result through the flow with its typed metadata attached', async () => {
		mockSnapshot.mockResolvedValue(buildSkillRunSnapshot('/run-cwd', catalogOf('demo-skill')));
		mockRead.mockResolvedValue(resourceResult('demo-skill', 'refs/DETAILS.md'));
		mockRecordActivation.mockResolvedValue({
			created: false,
			extra: skillResourceExtra(resourceResult('demo-skill', 'refs/DETAILS.md')),
			toolResultMessage: null
		});
		mockToolCallTurn(resourceSkillToolCallJson());

		const { callbacks, createToolResultMessage } = makeCallbacks();
		const runPromise = agenticStore.runAgenticFlow(runParams('conv-1', callbacks));

		await waitForPermission('conv-1');
		agenticStore.resolvePermission('conv-1', ToolPermissionDecision.ONCE);

		await runPromise;

		expect(mockRecordActivation).toHaveBeenCalled();
		expect(createToolResultMessage).toHaveBeenCalledWith(
			'call_1',
			'<skill_resource name="demo-skill" path="refs/DETAILS.md">data</skill_resource>',
			[expect.objectContaining({ kind: 'resource', path: 'refs/DETAILS.md' })]
		);
	});
});
