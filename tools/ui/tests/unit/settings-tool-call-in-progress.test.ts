// Guards the Settings > Display wiring of "Show tool call in progress": the
// flat config default and its presence as a checkbox in the Display section
// come from the settings registry, so the key/label/type/section pairing is the
// contract the dialog renders.

import {
	SETTING_CONFIG_DEFAULT,
	SETTINGS_CHAT_SECTIONS,
	SETTINGS_KEYS,
	SETTINGS_SECTION_SLUGS
} from '$lib/constants';
import { SettingsFieldType } from '$lib/enums';
import { describe, expect, it } from 'vitest';

function displayField(key: string) {
	const section = SETTINGS_CHAT_SECTIONS.find((s) => s.slug === SETTINGS_SECTION_SLUGS.DISPLAY);

	return section?.fields?.find((field) => field?.key === key);
}

describe('showToolCallInProgress setting', () => {
	it('is a checkbox in the Display section', () => {
		const field = displayField(SETTINGS_KEYS.SHOW_TOOL_CALL_IN_PROGRESS);

		expect(field).toBeDefined();
		expect(field?.type).toBe(SettingsFieldType.CHECKBOX);
		expect(field?.label).toBe('Show tool call in progress');
	});

	it('ships enabled, like "Show thought in progress"', () => {
		expect(SETTING_CONFIG_DEFAULT[SETTINGS_KEYS.SHOW_TOOL_CALL_IN_PROGRESS]).toBe(true);
		expect(SETTING_CONFIG_DEFAULT[SETTINGS_KEYS.SHOW_THOUGHT_IN_PROGRESS]).toBe(true);
	});

	it('is listed next to its "always show" counterpart', () => {
		const section = SETTINGS_CHAT_SECTIONS.find(
			(s) => s.slug === SETTINGS_SECTION_SLUGS.DISPLAY
		) as (typeof SETTINGS_CHAT_SECTIONS)[number];
		const keys = section.fields?.map((field) => field?.key) ?? [];
		const inProgressIndex = keys.indexOf(SETTINGS_KEYS.SHOW_TOOL_CALL_IN_PROGRESS);

		expect(inProgressIndex).toBeGreaterThanOrEqual(0);
		expect(keys[inProgressIndex + 1]).toBe(SETTINGS_KEYS.ALWAYS_SHOW_TOOL_CALL_CONTENT);
	});
});
