import {
	buildSkillListToolDefinition,
	buildSkillReadToolDefinition,
	SKILL_LIST_TOOL,
	SKILL_READ_TOOL,
	SKILL_SERVER_LABEL,
	SKILL_TOOL_SETTINGS
} from '$lib/constants';
import { describe, expect, it } from 'vitest';

describe('Skills tool registry names and settings keys', () => {
	it('exposes stable model-facing names', () => {
		expect(SKILL_READ_TOOL).toBe('read_skill');
		expect(SKILL_LIST_TOOL).toBe('list_skill');
		expect(SKILL_SERVER_LABEL).toBe('Skills');
	});

	it('exposes one settings row per adapter with stable keys and labels', () => {
		expect(SKILL_TOOL_SETTINGS).toHaveLength(2);

		const byName = new Map(SKILL_TOOL_SETTINGS.map((setting) => [setting.toolName, setting]));

		expect(byName.get(SKILL_READ_TOOL)).toMatchObject({
			key: 'skill:read_skill',
			label: 'Read skill',
			toolName: 'read_skill'
		});
		expect(byName.get(SKILL_LIST_TOOL)).toMatchObject({
			key: 'skill:list_skill',
			label: 'List skills',
			toolName: 'list_skill'
		});
	});

	it('keeps settings keys distinct from generic builtin/MCP/custom selection keys', () => {
		for (const setting of SKILL_TOOL_SETTINGS) {
			expect(setting.key.startsWith('skill:')).toBe(true);
			expect(setting.key.endsWith(setting.toolName)).toBe(true);
		}
	});

	it('provides display metadata and a renderable icon for each settings row', () => {
		for (const setting of SKILL_TOOL_SETTINGS) {
			expect(setting.label.length).toBeGreaterThan(0);
			expect(setting.description.length).toBeGreaterThan(0);
			expect(typeof setting.icon).toBe('function');
		}
	});
});

describe('buildSkillListToolDefinition', () => {
	it('is a no-argument list_skill definition', () => {
		const def = buildSkillListToolDefinition();

		expect(def.type).toBe('function');
		expect(def.function.name).toBe(SKILL_LIST_TOOL);
		expect(def.function.description).toBe(
			'List the skills available in this run, with their descriptions.'
		);
		expect(def.function.parameters).toEqual({
			properties: {},
			required: [],
			type: 'object'
		});
	});
});

describe('buildSkillReadToolDefinition', () => {
	it('requires name and keeps path optional in the display form', () => {
		const def = buildSkillReadToolDefinition();

		expect(def.type).toBe('function');
		expect(def.function.name).toBe(SKILL_READ_TOOL);
		expect(def.function.parameters).toMatchObject({
			properties: {
				name: { type: 'string' },
				path: { type: 'string' }
			},
			required: ['name'],
			type: 'object'
		});
	});

	it('adds the dynamic snapshot name enum only when names are supplied', () => {
		const run = buildSkillReadToolDefinition(['alpha', 'beta']);
		const nameParam = (run.function.parameters.properties as { name: { enum?: string[] } }).name;

		expect(nameParam.enum).toEqual(['alpha', 'beta']);
	});

	it('never carries a static name enum in the no-argument display form', () => {
		const display = buildSkillReadToolDefinition();
		const nameParam = (display.function.parameters.properties as { name: { enum?: unknown } }).name;

		expect(nameParam.enum).toBeUndefined();
	});

	it('copies the supplied names so later caller mutation cannot leak into a run definition', () => {
		const names = ['alpha'];
		const run = buildSkillReadToolDefinition(names);

		names.push('beta');
		const nameParam = (run.function.parameters.properties as { name: { enum?: string[] } }).name;

		expect(nameParam.enum).toEqual(['alpha']);
	});
});

describe('SKILL_TOOL_SETTINGS definitions', () => {
	it('reuses the no-argument builders for the settings-only display definitions', () => {
		const byKey = new Map(SKILL_TOOL_SETTINGS.map((setting) => [setting.key, setting]));

		expect(byKey.get('skill:read_skill')?.definition).toEqual(buildSkillReadToolDefinition());
		expect(byKey.get('skill:list_skill')?.definition).toEqual(buildSkillListToolDefinition());
	});

	it('freezes the settings-only definitions so they cannot be mutated into run definitions', () => {
		for (const setting of SKILL_TOOL_SETTINGS) {
			expect(Object.isFrozen(setting.definition)).toBe(true);
			expect(Object.isFrozen(setting.definition.function)).toBe(true);
			expect(Object.isFrozen(setting.definition.function.parameters)).toBe(true);
		}
	});

	it('does not leak a snapshot enum into any settings-only definition', () => {
		for (const setting of SKILL_TOOL_SETTINGS) {
			const parameters = setting.definition.function.parameters;
			const properties = parameters.properties as Record<string, { enum?: unknown }> | undefined;

			if (!properties) continue;

			for (const prop of Object.values(properties)) {
				expect(prop.enum).toBeUndefined();
			}
		}
	});
});
