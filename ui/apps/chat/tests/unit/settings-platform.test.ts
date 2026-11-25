import { SETTINGS_CHAT_SECTIONS, SETTINGS_KEYS } from '#lib/constants/index.ts';
import { SettingsPlatform } from '#lib/enums/index.ts';
import { isSettingShownOn } from '#lib/utils/index.ts';
import { describe, expect, it } from 'vitest';

const fields = SETTINGS_CHAT_SECTIONS.flatMap((section) => section.fields ?? []);

function shownKeys(isMobile: boolean): string[] {
	return fields.filter((field) => isSettingShownOn(field, isMobile)).map((field) => field.key);
}

describe('settings platform', () => {
	it('lists a setting without a platform everywhere', () => {
		expect(isSettingShownOn({}, true)).toBe(true);
		expect(isSettingShownOn({ platform: SettingsPlatform.ALL }, false)).toBe(true);
	});

	it('leaves desktop settings off a phone', () => {
		const onPhone = shownKeys(true);

		for (const key of [
			SETTINGS_KEYS.ALWAYS_SHOW_SIDEBAR_ON_DESKTOP,
			SETTINGS_KEYS.CONVERSATION_TABS,
			SETTINGS_KEYS.SINGLE_ROW_CHAT_FORM,
			SETTINGS_KEYS.SEND_ON_ENTER
		]) {
			expect(onPhone).not.toContain(key);
			expect(shownKeys(false)).toContain(key);
		}
	});

	it('leaves phone settings off a desktop', () => {
		expect(shownKeys(false)).not.toContain(SETTINGS_KEYS.SHOW_MODEL_ORG_NAME_ON_MOBILE);
		expect(shownKeys(true)).toContain(SETTINGS_KEYS.SHOW_MODEL_ORG_NAME_ON_MOBILE);
	});
});
