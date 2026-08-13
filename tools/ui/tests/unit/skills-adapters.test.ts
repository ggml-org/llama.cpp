import { buildSkillRunSnapshot } from '$lib/services/skills-packing.service';
import {
	SKILL_LIST_TOOL,
	SKILL_READ_TOOL,
	buildSkillToolDefinitions,
	consentKeyFor,
	decorateSkillPrompt,
	listSkillContent,
	skillDenialResult,
	skillErrorResult
} from '$lib/services/skills-adapters.service';
import type {
	SkillAdaptersBuildResult,
	SkillCatalogEntry,
	SkillCatalogResponse,
	SkillPackedCatalog
} from '$lib/services/skills-adapters.service';
import { MessageRole } from '$lib/enums';
import { describe, expect, it } from 'vitest';

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

function packed(overrides: Partial<SkillPackedCatalog>): SkillPackedCatalog {
	return {
		envelope:
			'<skills_catalog total="1" included="1"><available_skills>instr</available_skills><skill><name>alpha</name></skill></skills_catalog>',
		total: 1,
		included: 1,
		estimated: true,
		...overrides
	};
}

describe('buildSkillToolDefinitions', () => {
	it('registers no adapters and no diagnostics for a zero-budget or empty envelope', () => {
		const snapshot = buildSkillRunSnapshot('/cwd', makeCatalog('alpha'));

		const result = buildSkillToolDefinitions(snapshot, packed({ envelope: '' }), new Set());

		expect(result).toEqual({ definitions: [], diagnostics: [] });
	});

	it('registers only read_skill for a complete envelope', () => {
		const snapshot = buildSkillRunSnapshot('/cwd', makeCatalog('alpha', 'beta'));
		const complete = packed({
			envelope: '<skills_catalog total="2" included="2">...</skills_catalog>',
			included: 2,
			total: 2
		});

		const { definitions, diagnostics } = buildSkillToolDefinitions(snapshot, complete, new Set());

		expect(diagnostics).toEqual([]);
		expect(definitions.map((d) => d.function.name)).toEqual([SKILL_READ_TOOL]);
	});

	it('registers read_skill plus list_skill for a partial envelope', () => {
		const snapshot = buildSkillRunSnapshot('/cwd', makeCatalog('alpha', 'beta'));
		const partial = packed({
			envelope: '<skills_catalog total="2" included="1">...</skills_catalog>',
			included: 1,
			total: 2
		});

		const { definitions } = buildSkillToolDefinitions(snapshot, partial, new Set());

		expect(definitions.map((d) => d.function.name)).toEqual([SKILL_READ_TOOL, SKILL_LIST_TOOL]);
	});

	it('constrains read_skill name to frozen snapshot names via a dynamic enum and keeps path optional', () => {
		const snapshot = buildSkillRunSnapshot('/cwd', makeCatalog('alpha', 'beta'));

		const { definitions } = buildSkillToolDefinitions(snapshot, packed({ total: 2 }), new Set());

		const readSkill = definitions.find((d) => d.function.name === SKILL_READ_TOOL)!;

		expect(readSkill.function.parameters).toMatchObject({
			type: 'object',
			required: ['name'],
			properties: {
				name: { type: 'string', enum: ['alpha', 'beta'] },
				path: { type: 'string' }
			}
		});
	});

	it('derives the enum from the snapshot entries, never from the caller', () => {
		const snapshot = buildSkillRunSnapshot('/cwd', makeCatalog('alpha'));

		const first = buildSkillToolDefinitions(snapshot, packed({ total: 1 }), new Set());
		const second = buildSkillToolDefinitions(snapshot, packed({ total: 1 }), new Set());

		const nameParam = (defs: SkillAdaptersBuildResult['definitions']) =>
			defs.find((d) => d.function.name === SKILL_READ_TOOL)!.function.parameters.properties
				.name as { enum: string[] };

		expect(nameParam(first.definitions)).toEqual({ enum: ['alpha'], type: 'string' });
		expect(nameParam(second.definitions)).toEqual({ enum: ['alpha'], type: 'string' });
	});

	it('omits colliding adapters in favor of existing non-Skills tools with a safe diagnostic', () => {
		const snapshot = buildSkillRunSnapshot('/cwd', makeCatalog('alpha'));

		const { definitions, diagnostics } = buildSkillToolDefinitions(
			snapshot,
			packed({}),
			new Set([SKILL_READ_TOOL, 'other_tool'])
		);

		expect(definitions.map((d) => d.function.name)).toEqual([]);
		expect(diagnostics).toEqual([
			{
				code: 'skill_adapter_collision',
				message: `Skills tool "${SKILL_READ_TOOL}" collides with an existing tool and was not registered.`,
				name: SKILL_READ_TOOL
			}
		]);
	});

	it('keeps the non-colliding adapter when only one name collides', () => {
		const snapshot = buildSkillRunSnapshot('/cwd', makeCatalog('alpha'));
		const partial = packed({ envelope: '<skills_catalog total="1" included="0">i</skills_catalog>', included: 0 });

		const { definitions, diagnostics } = buildSkillToolDefinitions(snapshot, partial, new Set([SKILL_LIST_TOOL]));

		expect(definitions.map((d) => d.function.name)).toEqual([SKILL_READ_TOOL]);
		expect(diagnostics).toHaveLength(1);
		expect(diagnostics[0].name).toBe(SKILL_LIST_TOOL);
	});

	it('returns immutable definitions', () => {
		const snapshot = buildSkillRunSnapshot('/cwd', makeCatalog('alpha'));

		const { definitions } = buildSkillToolDefinitions(snapshot, packed({}), new Set());

		for (const def of definitions) {
			expect(Object.isFrozen(def)).toBe(true);
			expect(Object.isFrozen(def.function)).toBe(true);
			expect(Object.isFrozen(def.function.parameters)).toBe(true);
		}
	});
});

describe('decorateSkillPrompt', () => {
	const envelope =
		'<skills_catalog total="1" included="1"><available_skills>Call read_skill(name) when matching.</available_skills><skill><name>alpha</name></skill></skills_catalog>';

	it('appends the envelope byte-for-byte to the first system message', () => {
		const messages = [
			{ content: 'You are a helpful assistant.', role: MessageRole.SYSTEM },
			{ content: 'hi', role: MessageRole.USER }
		];

		const decorated = decorateSkillPrompt(messages, envelope);

		expect(decorated).toHaveLength(2);
		expect(decorated[0].content).toContain('You are a helpful assistant.');
		expect(decorated[0].content).toContain(envelope);
		expect(decorated[0].content).not.toBe(envelope);
	});

	it('prepends a system message when the run has no system message', () => {
		const messages = [{ content: 'hi', role: MessageRole.USER }];

		const decorated = decorateSkillPrompt(messages, envelope);

		expect(decorated).toHaveLength(2);
		expect(decorated[0]).toEqual({ content: envelope, role: MessageRole.SYSTEM });
		expect(decorated[1]).toEqual(messages[0]);
	});

	it('does not mutate the input messages', () => {
		const messages = [
			{ content: 'You are a helpful assistant.', role: MessageRole.SYSTEM },
			{ content: 'hi', role: MessageRole.USER }
		];
		const snapshot = structuredClone(messages);

		decorateSkillPrompt(messages, envelope);

		expect(messages).toEqual(snapshot);
	});

	it('leaves messages untouched when the envelope is empty', () => {
		const messages = [{ content: 'hi', role: MessageRole.USER }];

		expect(decorateSkillPrompt(messages, '')).toBe(messages);
	});

	it('never re-escapes or parses the envelope XML', () => {
		const tricky =
			'<skills_catalog total="1" included="1"><skill_content name="a&amp;b">&lt;script&gt;alert(1)&lt;/script&gt;</skill_content></skills_catalog>';
		const messages = [{ content: 'You are a helpful assistant.', role: MessageRole.SYSTEM }];

		const decorated = decorateSkillPrompt(messages, tricky);

		expect(decorated[0].content).toContain(tricky);
		expect(decorated[0].content).not.toContain('&amp;amp;');
	});
});

describe('listSkillContent', () => {
	it('returns structured snapshot entries only, never XML', () => {
		const content = listSkillContent(buildSkillRunSnapshot('/cwd', makeCatalog('alpha', 'beta')).entries);

		expect(JSON.parse(content)).toEqual([
			{
				name: 'alpha',
				description: 'description of alpha',
				scope: 'project',
				provider: 'agents'
			},
			{
				name: 'beta',
				description: 'description of beta',
				scope: 'project',
				provider: 'agents'
			}
		]);
		expect(content).not.toContain('<skill>');
		expect(content).not.toContain('opaque-');
	});
});

describe('consentKeyFor', () => {
	it('treats the same opaque id under different CWDs as distinct consent identities', () => {
		expect(consentKeyFor('/a', 'opaque-1')).not.toBe(consentKeyFor('/b', 'opaque-1'));
		expect(consentKeyFor(undefined, 'opaque-1')).not.toBe(consentKeyFor('/a', 'opaque-1'));
	});

	it('is stable for the same CWD and id', () => {
		expect(consentKeyFor('/a', 'opaque-1')).toBe(consentKeyFor('/a', 'opaque-1'));
	});
});

describe('structured skill results', () => {
	it('builds a structured denial with no XML content', () => {
		const content = skillDenialResult(SKILL_READ_TOOL);

		expect(JSON.parse(content)).toEqual({
			message: 'Skill access was denied by the user.',
			status: 'denied',
			tool: SKILL_READ_TOOL
		});
		expect(content).not.toContain('<');
	});

	it('builds a structured error with no XML content', () => {
		const content = skillErrorResult(SKILL_READ_TOOL, 'boom');

		expect(JSON.parse(content)).toEqual({
			message: 'boom',
			status: 'error',
			tool: SKILL_READ_TOOL
		});
	});
});
