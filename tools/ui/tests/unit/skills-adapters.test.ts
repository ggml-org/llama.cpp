// Guards adapter registration, prompt decoration, and the consent/activation core.
import { baseResult, catalogOf, resourceResult } from '../fixtures/skills';
import { SKILL_LIST_TOOL, SKILL_READ_TOOL, SKILL_SERVER_LABEL } from '$lib/constants';
import { MessageRole, ToolCallType, ToolPermissionDecision } from '$lib/enums';
import * as SkillsServiceModule from '$lib/services/skills.service';
import { SkillsService } from '$lib/services/skills.service';
import { buildSkillRunSnapshot } from '$lib/services/skills.service';
import { skillActivationExtra, skillResourceExtra } from '$lib/services/skills-activation.service';
import type {
	SkillActivationInput,
	SkillActivationStore,
	SkillRunAdaptersOptions
} from '$lib/services/skills-adapters.service';
import {
	buildSkillToolDefinitions,
	decorateSkillPrompt,
	listSkillContent,
	skillDenialResult,
	skillErrorResult,
	SkillRunAdapters
} from '$lib/services/skills-adapters.service';
import type { SkillPackedCatalog } from '$lib/types';
import type { DatabaseMessage } from '$lib/types';
import type { AgenticToolCallPayload } from '$lib/types/agentic';
import type { Mock } from 'vitest';
import { beforeEach, describe, expect, it, vi } from 'vitest';

vi.mock('$lib/services/skills.service', async (importOriginal) => {
	const actual = await importOriginal<typeof SkillsServiceModule>();

	return {
		...actual,
		SkillsService: { list: vi.fn(), read: vi.fn() }
	};
});

const mockRead = vi.mocked(SkillsService.read);

type PermissionFn = (...args: unknown[]) => Promise<ToolPermissionDecision>;
type PermissionMock = Mock<PermissionFn>;

function packed(overrides: Partial<SkillPackedCatalog>): SkillPackedCatalog {
	return {
		envelope:
			'<skills_catalog total="1" included="1"><available_skills>instr</available_skills><skill><name>alpha</name></skill></skills_catalog>',
		estimated: true,
		included: 1,
		total: 1,
		...overrides,
		fullTokens: overrides.fullTokens ?? null
	};
}

const PARTIAL_ENVELOPE =
	'<skills_catalog total="1" included="0"><available_skills>instr</available_skills></skills_catalog>';

function defaultPermission(): PermissionMock {
	const mock = vi.fn<PermissionFn>();

	mock.mockResolvedValue(ToolPermissionDecision.ONCE);

	return mock;
}

/** Minimal durable-store double for activation routing tests. */
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
	const snapshot = buildSkillRunSnapshot(options.cwd, catalogOf(...names));
	const packedCatalog =
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
			{ function: { name: SKILL_READ_TOOL, parameters: {} }, type: 'function' },
			{ function: { name: SKILL_LIST_TOOL, parameters: {} }, type: 'function' }
		],
		packed: packedCatalog,
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

describe('buildSkillToolDefinitions', () => {
	it.each([
		['empty envelope', { envelope: '' }, new Set<string>(), undefined, [], []],
		[
			'complete envelope',
			{ included: 2, total: 2 },
			new Set<string>(),
			undefined,
			[SKILL_READ_TOOL],
			[]
		],
		[
			'partial envelope',
			{ included: 1, total: 2 },
			new Set<string>(),
			undefined,
			[SKILL_READ_TOOL, SKILL_LIST_TOOL],
			[]
		],
		[
			'enabled set keeps both',
			{ included: 1, total: 2 },
			new Set<string>(),
			[SKILL_READ_TOOL, SKILL_LIST_TOOL],
			[SKILL_READ_TOOL, SKILL_LIST_TOOL],
			[]
		],
		[
			'enabled set omits read_skill',
			{ included: 1, total: 2 },
			new Set<string>(),
			[SKILL_LIST_TOOL],
			[SKILL_LIST_TOOL],
			[]
		],
		[
			'enabled set omits list_skill',
			{ included: 1, total: 2 },
			new Set<string>(),
			[SKILL_READ_TOOL],
			[SKILL_READ_TOOL],
			[]
		],
		['empty enabled set', { included: 1, total: 2 }, new Set<string>(), [], [], []],
		['existing tool collides', {}, new Set([SKILL_READ_TOOL]), undefined, [], [SKILL_READ_TOOL]],
		[
			'disabled adapter collides',
			{ included: 0 },
			new Set([SKILL_LIST_TOOL]),
			[SKILL_READ_TOOL],
			[SKILL_READ_TOOL],
			[SKILL_LIST_TOOL]
		]
	] as const)(
		'registers adapters for %s',
		(_label, packedOverrides, existingTools, enabledNames, expectedNames, expectedDiagnostics) => {
			const enabled = enabledNames === undefined ? undefined : new Set(enabledNames);
			const snapshot = buildSkillRunSnapshot('/cwd', catalogOf('alpha'));
			const complete = packed({
				...packedOverrides,
				envelope:
					(packedOverrides as { envelope?: string }).envelope ??
					'<skills_catalog total="2" included="1">...</skills_catalog>'
			});
			const { definitions, diagnostics } = buildSkillToolDefinitions(
				snapshot,
				complete,
				existingTools,
				enabled
			);

			expect(definitions.map((d) => d.function.name)).toEqual([...expectedNames]);
			expect(diagnostics.map((d) => d.name)).toEqual([...expectedDiagnostics]);

			if (expectedDiagnostics.length > 0) {
				expect(diagnostics[0]).toMatchObject({ code: 'skill_adapter_collision' });
			}
		}
	);

	it('constrains read_skill name to frozen snapshot names via a dynamic enum and keeps path optional', () => {
		const snapshot = buildSkillRunSnapshot('/cwd', catalogOf('alpha', 'beta'));
		const { definitions } = buildSkillToolDefinitions(snapshot, packed({ total: 2 }), new Set());
		const readSkill = definitions.find((d) => d.function.name === SKILL_READ_TOOL)!;

		expect(readSkill.function.parameters).toMatchObject({
			properties: {
				name: { enum: ['alpha', 'beta'], type: 'string' },
				path: { type: 'string' }
			},
			required: ['name'],
			type: 'object'
		});
	});
});

describe('decorateSkillPrompt', () => {
	const envelope =
		'<skills_catalog total="1" included="1"><available_skills>Call read_skill(name) when matching.</available_skills><skill><name>alpha</name></skill></skills_catalog>';

	it('appends the envelope byte-for-byte to the first system message, or prepends one, without mutating the input', () => {
		const withSystem = [
			{ content: 'You are a helpful assistant.', role: MessageRole.SYSTEM },
			{ content: 'hi', role: MessageRole.USER }
		];
		const withoutSystem = [{ content: 'hi', role: MessageRole.USER }];
		const withSystemSnapshot = structuredClone(withSystem);
		const decoratedWith = decorateSkillPrompt(withSystem, envelope);
		const decoratedWithout = decorateSkillPrompt(withoutSystem, envelope);

		expect(decoratedWith).toHaveLength(2);
		expect(decoratedWith[0].content).toContain('You are a helpful assistant.');
		expect(decoratedWith[0].content).toContain(envelope);
		expect(decoratedWithout[0]).toEqual({ content: envelope, role: MessageRole.SYSTEM });
		expect(decoratedWithout[1]).toEqual(withoutSystem[0]);
		expect(withSystem).toEqual(withSystemSnapshot);
	});

	it('leaves messages untouched for an empty envelope and never re-escapes the XML', () => {
		const messages = [{ content: 'hi', role: MessageRole.USER }];
		const tricky =
			'<skills_catalog total="1" included="1"><skill_content name="a&amp;b">&lt;script&gt;alert(1)&lt;/script&gt;</skill_content></skills_catalog>';
		const decorated = decorateSkillPrompt(
			[{ content: 'You are a helpful assistant.', role: MessageRole.SYSTEM }],
			tricky
		);

		expect(decorateSkillPrompt(messages, '')).toBe(messages);
		expect(decorated[0].content).toContain(tricky);
		expect(decorated[0].content).not.toContain('&amp;amp;');
	});
});

describe('listSkillContent and structured results', () => {
	it('returns structured snapshot entries only, never XML or opaque IDs', () => {
		const content = listSkillContent(
			buildSkillRunSnapshot('/cwd', catalogOf('alpha', 'beta')).entries
		);

		expect(JSON.parse(content)).toEqual([
			{ description: 'description of alpha', name: 'alpha', provider: 'agents', scope: 'project' },
			{ description: 'description of beta', name: 'beta', provider: 'agents', scope: 'project' }
		]);
		expect(content).not.toContain('<skill>');
		expect(content).not.toContain('opaque-');
	});

	it('builds structured denial and error results with no XML content', () => {
		expect(JSON.parse(skillDenialResult(SKILL_READ_TOOL))).toEqual({
			message: 'Skill access was denied by the user.',
			status: 'denied',
			tool: SKILL_READ_TOOL
		});
		expect(JSON.parse(skillErrorResult(SKILL_READ_TOOL, 'boom'))).toEqual({
			message: 'boom',
			status: 'error',
			tool: SKILL_READ_TOOL
		});
	});
});

describe('SkillRunAdapters', () => {
	beforeEach(() => {
		mockRead.mockReset();
	});

	it('routes recognized tools with only the snapshot name/path and the snapshot CWD', async () => {
		expect(makeAdapters({}).isSkillTool(SKILL_READ_TOOL)).toBe(true);
		expect(makeAdapters({}).isSkillTool(SKILL_LIST_TOOL)).toBe(true);
		expect(makeAdapters({}).isSkillTool('not_a_skill_tool')).toBe(false);

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

	it('rejects malformed arguments, unknown names, and broken JSON without a server call', async () => {
		mockRead.mockResolvedValue(baseResult('demo-skill'));
		const adapters = makeAdapters({ names: ['demo-skill'] });
		const cases = [
			'{}',
			'{"name":"not-in-snapshot"}',
			'{"name":"demo-skill","path":7}',
			'{not json'
		];

		for (const args of cases) {
			const result = await adapters.execute({
				...readCall(SKILL_READ_TOOL),
				function: { arguments: args, name: SKILL_READ_TOOL }
			});

			expect(result.isError).toBe(true);
			expect(JSON.parse(result.content).status).toBe('error');
		}

		expect(mockRead).not.toHaveBeenCalled();
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

	it('pauses an unapproved identity, resumes on allow, and records the activation through the shared store', async () => {
		const contentXml =
			'<skill_content name="a&amp;b">&lt;code&gt;x &lt; y&lt;/code&gt;&amp; trailing</skill_content>';

		mockRead.mockResolvedValue(baseResult('demo-skill', { content_xml: contentXml }));
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
		// Server XML is preserved byte-for-byte in the tool result.
		expect(result.content).toBe(contentXml);
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

	it('authorizes a resource read from the durable base activation without consent', async () => {
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

	it('runs the approval flow for a resource read of an unapproved identity, with a session-only record', async () => {
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
		expect(activation.inputs).toHaveLength(2);
		expect(activation.activatedIds.has('opaque-demo-skill')).toBe(true);
	});

	it('treats the same skill name under a changed CWD as a distinct opaque identity requiring its own approval', async () => {
		const activation = fakeStore();

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

		await adaptersA.execute(readCall(SKILL_READ_TOOL));

		expect(requestPermission).not.toHaveBeenCalled();

		const result = await adaptersB.execute(readCall(SKILL_READ_TOOL), new AbortController().signal);

		expect(requestPermission).toHaveBeenCalledTimes(1);
		expect(result.isError).toBe(false);
		expect(result.activationRecorded).toBe(true);
		expect(activation.inputs[0].result.skill.id).toBe('opaque-id-A');
		expect(activation.inputs[1].result.skill.id).toBe('opaque-id-B');
	});

	it('shares one pending decision across concurrent reads of the same identity, each read dispatching its own server request', async () => {
		mockRead.mockResolvedValue(baseResult('demo-skill'));
		let resolvePermission!: (decision: ToolPermissionDecision) => void;

		const requestPermission = defaultPermission();

		requestPermission.mockImplementation(
			() => new Promise<ToolPermissionDecision>((resolve) => (resolvePermission = resolve))
		);
		const activation = fakeStore();
		const adapters = makeAdapters({ activation, requestPermission });
		const first = adapters.execute(readCall(SKILL_READ_TOOL));
		const second = adapters.execute(readCall(SKILL_READ_TOOL));

		await new Promise((r) => setTimeout(r, 0));

		expect(requestPermission).toHaveBeenCalledTimes(1);
		expect(mockRead).toHaveBeenCalledTimes(2);

		resolvePermission(ToolPermissionDecision.ONCE);

		const [firstResult, secondResult] = await Promise.all([first, second]);

		expect(firstResult.content).toBe('<skill_content name="demo-skill">body</skill_content>');
		expect(secondResult.content).toBe('<skill_content name="demo-skill">body</skill_content>');
		expect(activation.inputs).toHaveLength(2);
		expect(firstResult.activationRecorded).toBe(true);
		expect(secondResult.activationRecorded).toBeUndefined();
	});
});
