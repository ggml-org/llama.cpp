import { SKILL_LIST_TOOL, SKILL_READ_TOOL, SKILL_SERVER_LABEL } from '$lib/constants';
import { MessageRole, ToolCallType, ToolPermissionDecision } from '$lib/enums';
import { SkillsService } from '$lib/services/skills.service';
import { skillActivationExtra, skillResourceExtra } from '$lib/services/skills-activation.service';
import type {
	SkillActivationInput,
	SkillActivationStore,
	SkillRunAdaptersOptions
} from '$lib/services/skills-adapters.service';
import { SkillRunAdapters } from '$lib/services/skills-adapters.service';
import { buildSkillRunSnapshot } from '$lib/services/skills-packing.service';
import type {
	SkillBaseReadResult,
	SkillCatalogEntry,
	SkillCatalogResponse,
	SkillPackedCatalog,
	SkillResourceReadResult
} from '$lib/types';
import type { DatabaseMessage } from '$lib/types';
import type { AgenticToolCallPayload } from '$lib/types/agentic';
import type { Mock } from 'vitest';
import { beforeEach, describe, expect, it, vi } from 'vitest';

vi.mock('$lib/services/skills.service', () => ({
	SkillsService: { list: vi.fn(), read: vi.fn() }
}));

const mockRead = vi.mocked(SkillsService.read);

type PermissionFn = (...args: unknown[]) => Promise<ToolPermissionDecision>;
type PermissionMock = Mock<PermissionFn>;

function makeEntry(name: string): SkillCatalogEntry {
	return {
		catalog_xml: `<skill><name>${name}</name></skill>`,
		description: `description of ${name}`,
		id: `opaque-${name}`,
		instruction: { bytes: 16, lines: 1, modified_at: null, tokens: 4, tokens_estimated: true },
		name,
		provider: 'agents',
		resources: { count: 0, truncated: false },
		scope: 'project'
	};
}

function makeCatalog(...names: string[]): SkillCatalogResponse {
	return {
		catalog_instruction_xml:
			'<available_skills>Call read_skill(name) when matching.</available_skills>',
		diagnostics: [],
		skills: names.map(makeEntry)
	};
}

const PARTIAL_ENVELOPE =
	'<skills_catalog total="1" included="0"><available_skills>instr</available_skills></skills_catalog>';

function baseResult(
	name: string,
	overrides: Partial<SkillBaseReadResult> = {}
): SkillBaseReadResult {
	return {
		body_markdown: '# Body',
		content_xml: `<skill_content name="${name}">body</skill_content>`,
		diagnostics: [],
		kind: 'skill',
		resources: { paths: [], truncated: false },
		skill: { id: `opaque-${name}`, name, provider: 'agents', scope: 'project' },
		source: `---\nname: ${name}\n---\n# Body`,
		...overrides
	};
}

function resourceResult(name: string, path: string): SkillResourceReadResult {
	return {
		content_xml: `<skill_resource name="${name}" path="${path}">data</skill_resource>`,
		diagnostics: [],
		kind: 'resource',
		resource: { path },
		skill: { id: `opaque-${name}`, name, provider: 'agents', scope: 'project' }
	};
}

function defaultPermission(): PermissionMock {
	const mock = vi.fn<PermissionFn>();

	mock.mockResolvedValue(ToolPermissionDecision.ONCE);

	return mock;
}

/**
 * Minimal durable-store double recording every activation input. Mirrors the
 * real store's contract: a base read with an unactivated identity creates a
 * record and returns the persisted tool result message; everything else
 * dedupes or is session-only.
 */
function fakeStore(): SkillActivationStore & {
	inputs: SkillActivationInput[];
	activatedIds: Set<string>;
} {
	const activatedIds = new Set<string>();
	const inputs: SkillActivationInput[] = [];

	return {
		activatedIds,
		inputs,
		isActivated: (_conversationId, identityId) => activatedIds.has(identityId),
		loadConversation: vi.fn().mockResolvedValue(undefined),
		recordActivation: vi.fn(async (input) => {
			inputs.push(input);

			const id = input.result.skill.id;

			if (input.result.kind === 'resource') {
				activatedIds.add(id);

				return {
					created: false,
					extra: skillResourceExtra(input.result),
					toolResultMessage: null
				};
			}

			if (activatedIds.has(id)) {
				return {
					created: false,
					extra: skillActivationExtra(input.result),
					toolResultMessage: null
				};
			}

			activatedIds.add(id);

			return {
				created: true,
				extra: skillActivationExtra(input.result),
				toolResultMessage: {
					id: 'recorded-tool-result',
					role: MessageRole.TOOL
				} as DatabaseMessage
			};
		})
	};
}

function makeAdapters(options: {
	names?: string[];
	cwd?: string;
	conversationId?: string;
	packed?: SkillPackedCatalog;
	requestPermission?: SkillRunAdaptersOptions['requestPermission'];
	activation?: SkillActivationStore;
}): SkillRunAdapters {
	const names = options.names ?? ['demo-skill'];
	const snapshot = buildSkillRunSnapshot(options.cwd, makeCatalog(...names));
	const packed =
		options.packed ??
		({
			envelope: PARTIAL_ENVELOPE,
			estimated: true,
			included: 0,
			total: names.length
		} as SkillPackedCatalog);

	return new SkillRunAdapters({
		activation: options.activation ?? fakeStore(),
		conversationId: options.conversationId ?? 'conv-1',
		definitions: [
			{
				function: { name: SKILL_READ_TOOL, parameters: {} },
				type: 'function'
			},
			{
				function: { name: SKILL_LIST_TOOL, parameters: {} },
				type: 'function'
			}
		],
		packed,
		requestPermission: options.requestPermission ?? defaultPermission(),
		snapshot
	});
}

function readCall(name: string, path?: string): AgenticToolCallPayload {
	const args =
		path !== undefined
			? JSON.stringify({ name: 'demo-skill', path })
			: JSON.stringify({ name: 'demo-skill' });

	return {
		function: { arguments: args, name },
		id: 'call_1',
		type: ToolCallType.FUNCTION
	};
}

describe('SkillRunAdapters', () => {
	beforeEach(() => {
		mockRead.mockReset();
	});

	it('recognizes only registered skill tool names', () => {
		const adapters = makeAdapters({});

		expect(adapters.isSkillTool(SKILL_READ_TOOL)).toBe(true);
		expect(adapters.isSkillTool(SKILL_LIST_TOOL)).toBe(true);
		expect(adapters.isSkillTool('not_a_skill_tool')).toBe(false);
	});

	it('list_skill returns structured snapshot entries without any server call or consent', async () => {
		const requestPermission = defaultPermission();

		requestPermission.mockResolvedValue(ToolPermissionDecision.DENY);
		const adapters = makeAdapters({ names: ['alpha', 'beta'], requestPermission });
		const result = await adapters.execute(readCall(SKILL_LIST_TOOL));

		expect(result.isError).toBe(false);
		expect(JSON.parse(result.content)).toEqual([
			{ description: 'description of alpha', name: 'alpha', provider: 'agents', scope: 'project' },
			{ description: 'description of beta', name: 'beta', provider: 'agents', scope: 'project' }
		]);
		expect(mockRead).not.toHaveBeenCalled();
		expect(requestPermission).not.toHaveBeenCalled();
	});

	it('sends only snapshot name/path through SkillsService with the snapshot CWD', async () => {
		mockRead.mockResolvedValue(baseResult('demo-skill'));
		const adapters = makeAdapters({ cwd: '/run-cwd' });
		const signal = new AbortController().signal;

		await adapters.execute(readCall(SKILL_READ_TOOL), signal);
		await adapters.execute(readCall(SKILL_READ_TOOL, 'refs/DETAILS.md'), signal);

		expect(mockRead).toHaveBeenNthCalledWith(1, { name: 'demo-skill' }, '/run-cwd', signal);
		expect(mockRead).toHaveBeenNthCalledWith(
			2,
			{ name: 'demo-skill', path: 'refs/DETAILS.md' },
			'/run-cwd',
			signal
		);
	});

	it('never dispatches a read for a name outside the snapshot', async () => {
		mockRead.mockResolvedValue(baseResult('demo-skill'));
		const adapters = makeAdapters({ names: ['demo-skill'] });
		const result = await adapters.execute({
			...readCall(SKILL_READ_TOOL),
			function: { arguments: JSON.stringify({ name: 'not-in-snapshot' }), name: SKILL_READ_TOOL }
		});

		expect(result.isError).toBe(true);
		expect(JSON.parse(result.content).status).toBe('error');
		expect(mockRead).not.toHaveBeenCalled();
	});

	it('rejects malformed arguments without a server call', async () => {
		const adapters = makeAdapters({});
		const missingName = await adapters.execute({
			...readCall(SKILL_READ_TOOL),
			function: { arguments: JSON.stringify({}), name: SKILL_READ_TOOL }
		});
		const badPath = await adapters.execute({
			...readCall(SKILL_READ_TOOL),
			function: {
				arguments: JSON.stringify({ name: 'demo-skill', path: 7 }),
				name: SKILL_READ_TOOL
			}
		});

		expect(missingName.isError).toBe(true);
		expect(badPath.isError).toBe(true);
		expect(mockRead).not.toHaveBeenCalled();
	});

	it('pauses an unapproved identity, resumes on allow, and records the activation through the shared store', async () => {
		mockRead.mockResolvedValue(baseResult('demo-skill'));
		const requestPermission = defaultPermission();
		const activation = fakeStore();
		const adapters = makeAdapters({
			activation,
			conversationId: 'conv-9',
			cwd: '/run-cwd',
			requestPermission
		});
		const result = await adapters.execute(readCall(SKILL_READ_TOOL), new AbortController().signal);

		expect(requestPermission).toHaveBeenCalledWith(
			SKILL_READ_TOOL,
			SKILL_SERVER_LABEL,
			{ name: 'demo-skill', provider: 'agents', scope: 'project' },
			expect.anything()
		);
		expect(result.isError).toBe(false);
		expect(result.content).toBe('<skill_content name="demo-skill">body</skill_content>');
		expect(result.activationRecorded).toBe(true);
		expect(result.recordedToolResultMessageId).toBe('recorded-tool-result');
		expect(result.extras).toHaveLength(1);
		expect(activation.inputs).toEqual([
			expect.objectContaining({
				conversationId: 'conv-9',
				cwd: '/run-cwd',
				toolCallId: 'call_1'
			})
		]);
	});

	it('returns a structured no-content denial on deny and records no activation', async () => {
		mockRead.mockResolvedValue(baseResult('demo-skill'));
		const requestPermission = defaultPermission();

		requestPermission.mockResolvedValue(ToolPermissionDecision.DENY);
		const activation = fakeStore();
		const adapters = makeAdapters({ activation, requestPermission });
		const result = await adapters.execute(readCall(SKILL_READ_TOOL));

		expect(result.isError).toBe(true);
		expect(JSON.parse(result.content)).toEqual({
			message: 'Skill access was denied by the user.',
			status: 'denied',
			tool: SKILL_READ_TOOL
		});
		expect(result.content).not.toContain('skill_content');
		expect(activation.inputs).toHaveLength(0);
		expect(activation.activatedIds.has('opaque-demo-skill')).toBe(false);
	});

	it('never consents and never records an activation on a failed server read', async () => {
		mockRead.mockRejectedValue(new Error('skills disabled'));
		const requestPermission = defaultPermission();
		const activation = fakeStore();
		const adapters = makeAdapters({ activation, requestPermission });
		const result = await adapters.execute(readCall(SKILL_READ_TOOL));

		expect(result.isError).toBe(true);
		expect(JSON.parse(result.content).status).toBe('error');
		expect(requestPermission).not.toHaveBeenCalled();
		expect(activation.inputs).toHaveLength(0);
	});

	it('authorizes a resource read from the durable base activation without consent and tags the result', async () => {
		const activation = fakeStore();

		activation.activatedIds.add('opaque-demo-skill');
		mockRead.mockResolvedValue(resourceResult('demo-skill', 'refs/DETAILS.md'));
		const requestPermission = defaultPermission();
		const adapters = makeAdapters({ activation, cwd: '/a', requestPermission });
		const result = await adapters.execute(readCall(SKILL_READ_TOOL, 'refs/DETAILS.md'));

		expect(result.isError).toBe(false);
		expect(result.content).toBe(
			'<skill_resource name="demo-skill" path="refs/DETAILS.md">data</skill_resource>'
		);
		expect(requestPermission).not.toHaveBeenCalled();
		expect(result.extras?.[0]).toMatchObject({ kind: 'resource', path: 'refs/DETAILS.md' });
	});

	it('runs the approval flow for a resource read of an unapproved identity, showing the requested path, with a session-only record', async () => {
		mockRead.mockResolvedValue(resourceResult('demo-skill', 'refs/DETAILS.md'));
		const requestPermission = defaultPermission();
		const activation = fakeStore();
		const adapters = makeAdapters({ activation, requestPermission });
		const result = await adapters.execute(
			readCall(SKILL_READ_TOOL, 'refs/DETAILS.md'),
			new AbortController().signal
		);

		expect(requestPermission).toHaveBeenCalledWith(
			SKILL_READ_TOOL,
			SKILL_SERVER_LABEL,
			{ name: 'demo-skill', path: 'refs/DETAILS.md', provider: 'agents', scope: 'project' },
			expect.anything()
		);
		expect(result.isError).toBe(false);
		expect(result.content).toBe(
			'<skill_resource name="demo-skill" path="refs/DETAILS.md">data</skill_resource>'
		);
		// Resource approvals are session-only: no durable record, the flow persists the message.
		expect(result.activationRecorded).toBeUndefined();
		expect(result.extras?.[0]).toMatchObject({ kind: 'resource' });
		expect(activation.inputs).toHaveLength(1);
	});

	it('does not re-prompt a second base read of an already-activated identity; the store dedupes the record', async () => {
		mockRead.mockResolvedValue(baseResult('demo-skill'));
		const requestPermission = defaultPermission();
		const activation = fakeStore();
		const adapters = makeAdapters({ activation, requestPermission });

		await adapters.execute(readCall(SKILL_READ_TOOL));
		await adapters.execute(readCall(SKILL_READ_TOOL));

		expect(requestPermission).toHaveBeenCalledTimes(1);
		expect(mockRead).toHaveBeenCalledTimes(2);
		// Both allowed reads route through the shared operation; the store
		// created exactly one durable record (the second call deduped).
		expect(activation.inputs).toHaveLength(2);
		expect(activation.activatedIds.has('opaque-demo-skill')).toBe(true);
	});

	it('treats the same skill name under a changed CWD as a distinct opaque identity requiring its own approval', async () => {
		const activation = fakeStore();

		// Durable activation of identity A (resolved under /a).
		activation.activatedIds.add('opaque-id-A');
		const requestPermission = defaultPermission();
		const adaptersA = makeAdapters({ activation, cwd: '/a', requestPermission });
		const adaptersB = makeAdapters({ activation, cwd: '/b', requestPermission });

		mockRead.mockResolvedValueOnce(
			baseResult('demo-skill', {
				skill: { id: 'opaque-id-A', name: 'demo-skill', provider: 'agents', scope: 'project' }
			})
		);
		mockRead.mockResolvedValueOnce(
			baseResult('demo-skill', {
				skill: { id: 'opaque-id-B', name: 'demo-skill', provider: 'agents', scope: 'project' }
			})
		);

		// The already-activated opaque id reads without any consent pause.
		await adaptersA.execute(readCall(SKILL_READ_TOOL));

		expect(requestPermission).not.toHaveBeenCalled();

		// The same skill name now resolves to a different opaque id (the
		// server resolved it under the changed CWD): the durable activation
		// of A does not authorize B, so B requires its own approval.
		const result = await adaptersB.execute(readCall(SKILL_READ_TOOL), new AbortController().signal);

		expect(requestPermission).toHaveBeenCalledTimes(1);
		expect(requestPermission).toHaveBeenCalledWith(
			SKILL_READ_TOOL,
			SKILL_SERVER_LABEL,
			{ name: 'demo-skill', provider: 'agents', scope: 'project' },
			expect.anything()
		);
		expect(result.isError).toBe(false);
		expect(result.content).toBe('<skill_content name="demo-skill">body</skill_content>');
		expect(result.activationRecorded).toBe(true);
		// Both allowed reads route through the shared operation: A deduped,
		// B created a new durable record under its own opaque id.
		expect(activation.inputs).toHaveLength(2);
		expect(activation.inputs[0].result.skill.id).toBe('opaque-id-A');
		expect(activation.inputs[1].result.skill.id).toBe('opaque-id-B');
	});

	it('shares one pending decision across concurrent base reads of the same resolved identity', async () => {
		mockRead.mockResolvedValue(baseResult('demo-skill'));
		let resolvePermission!: (decision: ToolPermissionDecision) => void;

		const requestPermission = defaultPermission();

		requestPermission.mockImplementation(
			() =>
				new Promise<ToolPermissionDecision>((resolve) => {
					resolvePermission = resolve;
				})
		);
		const activation = fakeStore();
		const adapters = makeAdapters({ activation, requestPermission });
		const first = adapters.execute(readCall(SKILL_READ_TOOL));
		const second = adapters.execute(readCall(SKILL_READ_TOOL));

		// Give both callers time to reach the pending decision before resolving.
		await new Promise((r) => setTimeout(r, 0));

		expect(requestPermission).toHaveBeenCalledTimes(1);

		resolvePermission(ToolPermissionDecision.ONCE);

		const [firstResult, secondResult] = await Promise.all([first, second]);

		expect(firstResult.content).toBe('<skill_content name="demo-skill">body</skill_content>');
		expect(secondResult.content).toBe('<skill_content name="demo-skill">body</skill_content>');
		// One durable record; the second read deduped through the store.
		expect(activation.inputs).toHaveLength(2);
		expect(firstResult.activationRecorded).toBe(true);
		expect(secondResult.activationRecorded).toBeUndefined();
	});

	it('does not deduplicate resource reads themselves, but shares their approval decision', async () => {
		mockRead.mockResolvedValue(resourceResult('demo-skill', 'refs/DETAILS.md'));
		let resolvePermission!: (decision: ToolPermissionDecision) => void;

		const requestPermission = defaultPermission();

		requestPermission.mockImplementation(
			() =>
				new Promise<ToolPermissionDecision>((resolve) => {
					resolvePermission = resolve;
				})
		);
		const adapters = makeAdapters({ requestPermission });
		const first = adapters.execute(readCall(SKILL_READ_TOOL, 'refs/DETAILS.md'));
		const second = adapters.execute(readCall(SKILL_READ_TOOL, 'refs/DETAILS.md'));

		await new Promise((r) => setTimeout(r, 0));

		// One prompt, but each read executed its own server request.
		expect(requestPermission).toHaveBeenCalledTimes(1);
		expect(mockRead).toHaveBeenCalledTimes(2);

		resolvePermission(ToolPermissionDecision.ONCE);

		const results = await Promise.all([first, second]);

		expect(results.every((r) => r.isError === false)).toBe(true);
	});

	it('preserves server XML byte-for-byte in allowed tool results', async () => {
		const contentXml =
			'<skill_content name="a&amp;b">&lt;code&gt;x &lt; y&lt;/code&gt;&amp; trailing</skill_content>';

		mockRead.mockResolvedValue(baseResult('demo-skill', { content_xml: contentXml }));
		const adapters = makeAdapters({});
		const result = await adapters.execute(readCall(SKILL_READ_TOOL));

		expect(result.isError).toBe(false);
		expect(result.content).toBe(contentXml);
	});

	it('rejects malformed tool call JSON with a structured error', async () => {
		const adapters = makeAdapters({});
		const result = await adapters.execute({
			...readCall(SKILL_READ_TOOL),
			function: { arguments: '{not json', name: SKILL_READ_TOOL }
		});

		expect(result.isError).toBe(true);
		expect(JSON.parse(result.content).status).toBe('error');
		expect(mockRead).not.toHaveBeenCalled();
	});

	it('routes the shared durable record with the model tool call id (activation reconstruction facts)', async () => {
		const activation = fakeStore();

		mockRead.mockResolvedValue(baseResult('demo-skill'));
		const adapters = makeAdapters({ activation, conversationId: 'conv-11' });

		await adapters.execute(readCall(SKILL_READ_TOOL));

		expect(activation.inputs[0]).toMatchObject({
			conversationId: 'conv-11',
			toolCallId: 'call_1'
		});
		expect(activation.inputs[0].result.skill.id).toBe('opaque-demo-skill');
	});
});
