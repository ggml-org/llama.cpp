import { describe, expect, it } from 'vitest';
import { SETTINGS_KEYS } from '$lib/constants/settings-keys';
import { SETTINGS_CHAT_SECTIONS } from '$lib/constants/settings-registry';

describe('checkApiKeyField', () => {
	it('should have isPrivate set to true', () => {
		const fields = SETTINGS_CHAT_SECTIONS.flatMap((section) => section.fields);
		const apiKeyField = fields.find((field) => field?.key === SETTINGS_KEYS.API_KEY);

		expect(apiKeyField).toBeDefined();
		expect(apiKeyField?.isPrivate).toBe(true);
	});
});
