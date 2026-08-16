// Guards the persisted Skills catalog budget contract: a non-negative
// integer defaulting to 2000, zero valid (it disables prompt packing, it
// does NOT mean the server catalog is empty), wired through the established
// numeric-control pattern (min 0 + integer rounding + load-time sanitize).

import {
	normalizeSkillBudget,
	POSITIVE_INTEGER_FIELDS,
	SETTING_CONFIG_DEFAULT,
	SETTINGS_CHAT_SECTIONS,
	SETTINGS_KEYS,
	SETTINGS_SECTION_SLUGS
} from '$lib/constants';
import { SettingsFieldType } from '$lib/enums/settings.enums';
import { describe, expect, it } from 'vitest';

describe('maxSkillBudget registry', () => {
	it('defaults to 2000 in the persisted config defaults', () => {
		expect(SETTING_CONFIG_DEFAULT[SETTINGS_KEYS.MAX_SKILL_BUDGET]).toBe(2000);
	});

	it('is registered as a non-negative integer input control in the agentic section', () => {
		const section = SETTINGS_CHAT_SECTIONS.find((s) => s.slug === SETTINGS_SECTION_SLUGS.AGENTIC);

		expect(section).toBeDefined();

		const field = section?.fields?.find((f) => f.key === SETTINGS_KEYS.MAX_SKILL_BUDGET);

		expect(field).toMatchObject({
			isPositiveInteger: true,
			min: 0,
			type: SettingsFieldType.INPUT
		});
	});

	it('participates in the save-time numeric clamp list', () => {
		expect(POSITIVE_INTEGER_FIELDS).toContain(SETTINGS_KEYS.MAX_SKILL_BUDGET);
	});
});

describe('normalizeSkillBudget', () => {
	it('keeps a valid non-negative integer untouched', () => {
		expect(normalizeSkillBudget(2000)).toBe(2000);
		expect(normalizeSkillBudget(0)).toBe(0);
	});

	it('clamps negative values to zero', () => {
		expect(normalizeSkillBudget(-5)).toBe(0);
		expect(normalizeSkillBudget(-0.1)).toBe(0);
	});

	it('rounds fractional values to integers', () => {
		expect(normalizeSkillBudget(3.7)).toBe(4);
		expect(normalizeSkillBudget(2500.2)).toBe(2500);
	});

	it('falls back to the default for non-numeric or non-finite values', () => {
		expect(normalizeSkillBudget('2500')).toBe(2000);
		expect(normalizeSkillBudget(undefined)).toBe(2000);
		expect(normalizeSkillBudget(null)).toBe(2000);
		expect(normalizeSkillBudget(Number.NaN)).toBe(2000);
		expect(normalizeSkillBudget(Number.POSITIVE_INFINITY)).toBe(2000);
	});
});
