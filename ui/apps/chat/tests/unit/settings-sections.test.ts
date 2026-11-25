import { SETTINGS_CHAT_SECTIONS, SETTINGS_SECTION_SLUGS } from '#lib/constants/index.ts';
import { groupSettingsFields, isApiKeyFieldShown } from '#lib/utils/index.ts';
import { describe, expect, it } from 'vitest';

describe('settings sections', () => {
	it('come in the navigation order', () => {
		expect(SETTINGS_CHAT_SECTIONS.map((section) => section.slug)).toEqual([
			SETTINGS_SECTION_SLUGS.GENERAL,
			SETTINGS_SECTION_SLUGS.CHAT,
			SETTINGS_SECTION_SLUGS.MODEL,
			SETTINGS_SECTION_SLUGS.AGENT,
			SETTINGS_SECTION_SLUGS.VOICE,
			SETTINGS_SECTION_SLUGS.DATA
		]);
	});

	it('put every field in a subgroup, each listed once per block', () => {
		for (const section of SETTINGS_CHAT_SECTIONS) {
			const fields = section.fields ?? [];

			expect(fields.every((field) => field.group)).toBe(true);

			for (const advanced of [false, true]) {
				const block = fields.filter((field) => Boolean(field.advanced) === advanced);
				const titles = groupSettingsFields(block).map((group) => group.title);

				expect(new Set(titles).size).toBe(titles.length);
			}
		}
	});

	it('list advanced fields after the others', () => {
		for (const section of SETTINGS_CHAT_SECTIONS) {
			const flags = (section.fields ?? []).map((field) => Boolean(field.advanced));

			expect(flags).toEqual([...flags].sort((a, b) => Number(a) - Number(b)));
		}
	});
});

describe('groupSettingsFields', () => {
	it('keeps registry order and splits on group changes', () => {
		const field = (key: string, group?: string) =>
			({ group, key, label: key }) as SettingsFieldConfig;

		expect(groupSettingsFields([field('a', 'X'), field('b', 'X'), field('c', 'Y')])).toEqual([
			{ fields: [field('a', 'X'), field('b', 'X')], title: 'X' },
			{ fields: [field('c', 'Y')], title: 'Y' }
		]);
	});
});

describe('isApiKeyFieldShown', () => {
	const none = { draftKey: '', status: 200, storedKey: '' };

	it('hides the field on a server that asks for no key', () => {
		expect(isApiKeyFieldShown(none)).toBe(false);
	});

	it('shows it for a stored or typed key, and when the server refuses the request', () => {
		expect(isApiKeyFieldShown({ ...none, storedKey: 'sk-1' })).toBe(true);
		expect(isApiKeyFieldShown({ ...none, draftKey: 's' })).toBe(true);
		expect(isApiKeyFieldShown({ ...none, status: 401 })).toBe(true);
		expect(isApiKeyFieldShown({ ...none, status: 403 })).toBe(true);
	});
});
