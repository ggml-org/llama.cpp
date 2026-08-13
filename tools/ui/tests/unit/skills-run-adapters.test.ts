import { buildSkillRunSnapshot } from '$lib/services/skills-packing.service';
import {
	SKILL_LIST_TOOL,
	SKILL_READ_TOOL,
	SKILL_SERVER_LABEL,
	SkillRunAdapters,
	consentKeyFor,
	createInMemorySkillActivationStore
} from '$lib/services/skills-adapters.service';
import { SkillsService } from '$lib/services/skills.service';
import type {
	SkillActivationStore,
	SkillRunAdaptersOptions
} from '$lib/services/skills-adapters.service';
import type {
	SkillBaseReadResult,
	SkillCatalogEntry,
	SkillCatalogResponse,
	SkillIdentity,
	SkillPackedCatalog
} from '$lib/types';
import { ToolPermissionDecision } from '$lib/enums';
import type { AgenticToolCallPayload } from '$lib/types/agentic';
import { beforeEach, describe, expect, it, vi } from 'vitest';

vi.mock('$lib/services/skills.service', () => ({
	SkillsService: { list: vi.fn(), read: vi.fn() }
}));

const mockRead = vi.mocked(SkillsService.read);

type PermissionMock = (...args: unknown[]) => Promise<ToolPermissionDecision>;

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

const PARTIAL_ENVELOPE =
	'<skills_catalog total="1" included="0"><available_skills>instr</available_skills></skills_catalog>';

function baseResult(name: string, overrides: Partial<SkillBaseReadResult> = {}): SkillBaseReadResult {
	return {
		kind: 'skill',
		skill: { id: `opaque-${name}`, name, scope: 'project', provider: 'agents' },
		resources: { paths: [], truncated: false },
		content_xml: `<skill_content name="${name}">body</skill_content>`,
		diagnostics: [],
		...overrides
	};
}

function defaultPermission(): PermissionMock {
	const mock = vi.fn<PermissionMock>();

	mock.mockResolvedValue(ToolPermissionDecision.ONCE);

	return mock;
}

function makeAdapters(options: {
	names?: string[];
	cwd?: string;
	packed?: SkillPackedCatalog;
	requestPermission?: SkillRunAdaptersOptions['requestPermission'];
	activation?: SkillActivationStore;
}): SkillRunAdapters {
	const names = options.names ?? ['demo-skill'];
	const snapshot = buildSkillRunSnapshot(options.cwd, makeCatalog(...names));
	const packed =
		options.packed ??
		({ envelope: PARTIAL_ENVELOPE, total: names.length, included: 0, estimated: true } as SkillPackedCatalog);

	return new SkillRunAdapters({
		snapshot,
		packed,
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
		requestPermission: options.requestPermission ?? defaultPermission(),
		...(options.activation ? { activation: options.activation } : {})
	});
}

function readCall(name: string, path?: string): AgenticToolCallPayload {
	const args =
		path !== undefined
			? JSON.stringify({ name: 'demo-skill', path })
			: JSON.stringify({ name: 'demo-skill' });

	return {
		id: 'call_1',
		type: 'function',
		function: { name, arguments: args }
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
			{ name: 'alpha', description: 'description of alpha', scope: 'project', provider: 'agents' },
			{ name: 'beta', description: 'description of beta', scope: 'project', provider: 'agents' }
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
			function: { name: SKILL_READ_TOOL, arguments: JSON.stringify({ name: 'not-in-snapshot' }) }
		});

		expect(result.isError).toBe(true);
		expect(JSON.parse(result.content).status).toBe('error');
		expect(mockRead).not.toHaveBeenCalled();
	});

	it('rejects malformed arguments without a server call', async () => {
		const adapters = makeAdapters({});

		const missingName = await adapters.execute({
			...readCall(SKILL_READ_TOOL),
			function: { name: SKILL_READ_TOOL, arguments: JSON.stringify({}) }
		});
		const badPath = await adapters.execute({
			...readCall(SKILL_READ_TOOL),
			function: { name: SKILL_READ_TOOL, arguments: JSON.stringify({ name: 'demo-skill', path: 7 }) }
		});

		expect(missingName.isError).toBe(true);
		expect(badPath.isError).toBe(true);
		expect(mockRead).not.toHaveBeenCalled();
	});

	it('pauses an unapproved resolved identity with safe identity metadata and resumes on allow with byte-preserved XML', async () => {
		mockRead.mockResolvedValue(baseResult('demo-skill'));
		const requestPermission = defaultPermission();
		const activation = createInMemorySkillActivationStore();
		const adapters = makeAdapters({ requestPermission, activation });

		const result = await adapters.execute(readCall(SKILL_READ_TOOL), new AbortController().signal);

		expect(requestPermission).toHaveBeenCalledWith(
			SKILL_READ_TOOL,
			SKILL_SERVER_LABEL,
			{ name: 'demo-skill', scope: 'project', provider: 'agents' },
			expect.anything()
		);
		expect(result.isError).toBe(false);
		expect(result.content).toBe('<skill_content name="demo-skill">body</skill_content>');
		expect(activation.isActivated(consentKeyFor(undefined, 'opaque-demo-skill'))).toBe(true);
	});

	it('returns a structured no-content denial on deny and records no activation', async () => {
		mockRead.mockResolvedValue(baseResult('demo-skill'));
		const requestPermission = defaultPermission();

		requestPermission.mockResolvedValue(ToolPermissionDecision.DENY);
		const activation = createInMemorySkillActivationStore();
		const adapters = makeAdapters({ requestPermission, activation });

		const result = await adapters.execute(readCall(SKILL_READ_TOOL));

		expect(result.isError).toBe(true);
		expect(JSON.parse(result.content)).toEqual({
			message: 'Skill access was denied by the user.',
			status: 'denied',
			tool: SKILL_READ_TOOL
		});
		expect(result.content).not.toContain('skill_content');
		expect(activation.isActivated(consentKeyFor(undefined, 'opaque-demo-skill'))).toBe(false);
	});

	it('never consents and never activates on a failed server read', async () => {
		mockRead.mockRejectedValue(new Error('skills disabled'));
		const requestPermission = defaultPermission();
		const activation = createInMemorySkillActivationStore();
		const adapters = makeAdapters({ requestPermission, activation });

		const result = await adapters.execute(readCall(SKILL_READ_TOOL));

		expect(result.isError).toBe(true);
		expect(JSON.parse(result.content).status).toBe('error');
		expect(requestPermission).not.toHaveBeenCalled();
		expect(activation.isActivated(consentKeyFor(undefined, 'opaque-demo-skill'))).toBe(false);
	});

	it('allows a resource read only after the exact resolved identity is activated', async () => {
		const activation = createInMemorySkillActivationStore();

		activation.recordActivation(consentKeyFor('/a', 'opaque-demo-skill'), {
			id: 'opaque-demo-skill',
			name: 'demo-skill',
			scope: 'project',
			provider: 'agents'
		});
		mockRead.mockResolvedValue({
			kind: 'resource',
			skill: { id: 'opaque-demo-skill', name: 'demo-skill', scope: 'project', provider: 'agents' },
			resource: { path: 'refs/DETAILS.md' },
			content_xml: '<skill_resource name="demo-skill" path="refs/DETAILS.md">data</skill_resource>',
			diagnostics: []
		});
		const requestPermission = defaultPermission();
		const adapters = makeAdapters({ cwd: '/a', requestPermission, activation });

		const result = await adapters.execute(readCall(SKILL_READ_TOOL, 'refs/DETAILS.md'));

		expect(result.isError).toBe(false);
		expect(result.content).toBe(
			'<skill_resource name="demo-skill" path="refs/DETAILS.md">data</skill_resource>'
		);
		expect(requestPermission).not.toHaveBeenCalled();
	});

	it('runs the approval flow for a resource read of an unapproved identity, showing the requested path', async () => {
		mockRead.mockResolvedValue({
			kind: 'resource',
			skill: { id: 'opaque-demo-skill', name: 'demo-skill', scope: 'project', provider: 'agents' },
			resource: { path: 'refs/DETAILS.md' },
			content_xml: '<skill_resource>data</skill_resource>',
			diagnostics: []
		});
		const requestPermission = defaultPermission();
		const activation = createInMemorySkillActivationStore();
		const adapters = makeAdapters({ requestPermission, activation });

		const result = await adapters.execute(readCall(SKILL_READ_TOOL, 'refs/DETAILS.md'), new AbortController().signal);

		expect(requestPermission).toHaveBeenCalledWith(
			SKILL_READ_TOOL,
			SKILL_SERVER_LABEL,
			{ name: 'demo-skill', scope: 'project', provider: 'agents', path: 'refs/DETAILS.md' },
			expect.anything()
		);
		expect(result.isError).toBe(false);
		expect(result.content).toBe('<skill_resource>data</skill_resource>');
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
		const adapters = makeAdapters({ requestPermission });

		const first = adapters.execute(readCall(SKILL_READ_TOOL));
		const second = adapters.execute(readCall(SKILL_READ_TOOL));

		// Give both callers time to reach the pending decision before resolving.
		await new Promise((r) => setTimeout(r, 0));

		expect(requestPermission).toHaveBeenCalledTimes(1);

		resolvePermission(ToolPermissionDecision.ONCE);

		const [firstResult, secondResult] = await Promise.all([first, second]);

		expect(firstResult.content).toBe('<skill_content name="demo-skill">body</skill_content>');
		expect(secondResult.content).toBe('<skill_content name="demo-skill">body</skill_content>');
	});

	it('does not deduplicate resource reads themselves, but shares their approval decision', async () => {
		mockRead.mockResolvedValue({
			kind: 'resource',
			skill: { id: 'opaque-demo-skill', name: 'demo-skill', scope: 'project', provider: 'agents' },
			resource: { path: 'refs/DETAILS.md' },
			content_xml: '<skill_resource>data</skill_resource>',
			diagnostics: []
		});
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

	it('requires a distinct consent identity when the CWD changes', async () => {
		const requestPermission = defaultPermission();

		mockRead.mockResolvedValue(baseResult('demo-skill'));

		const adaptersA = makeAdapters({ cwd: '/a', requestPermission });
		const adaptersB = makeAdapters({ cwd: '/b', requestPermission });

		await adaptersA.execute(readCall(SKILL_READ_TOOL));
		await adaptersB.execute(readCall(SKILL_READ_TOOL));

		// Same opaque id under a different CWD re-prompts (distinct consent key).
		expect(requestPermission).toHaveBeenCalledTimes(2);
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
			function: { name: SKILL_READ_TOOL, arguments: '{not json' }
		});

		expect(result.isError).toBe(true);
		expect(JSON.parse(result.content).status).toBe('error');
		expect(mockRead).not.toHaveBeenCalled();
	});

	it('does not re-prompt a second base read of an already-activated identity', async () => {
		mockRead.mockResolvedValue(baseResult('demo-skill'));
		const requestPermission = defaultPermission();
		const adapters = makeAdapters({ requestPermission });

		await adapters.execute(readCall(SKILL_READ_TOOL));
		await adapters.execute(readCall(SKILL_READ_TOOL));

		expect(requestPermission).toHaveBeenCalledTimes(1);
		expect(mockRead).toHaveBeenCalledTimes(2);
	});

	it('records activation through the injected activation store (Task 4 durable seam)', async () => {
		const recorded: { key: string; identity: SkillIdentity }[] = [];
		const activation: SkillActivationStore = {
			isActivated: () => false,
			recordActivation: (key, identity) => {
				recorded.push({ key, identity });
			}
		};

		mockRead.mockResolvedValue(baseResult('demo-skill'));
		const adapters = makeAdapters({ activation });

		await adapters.execute(readCall(SKILL_READ_TOOL));

		expect(recorded).toHaveLength(1);
		expect(recorded[0].key).toBe(consentKeyFor(undefined, 'opaque-demo-skill'));
		expect(recorded[0].identity.id).toBe('opaque-demo-skill');
	});
});
