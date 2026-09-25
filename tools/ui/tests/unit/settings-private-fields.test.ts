import { SETTINGS_CHAT_SECTIONS, SETTINGS_KEYS } from '$lib/constants';
import { describe, expect, it } from 'vitest';

describe('checkApiKeyField', () => {
	it('should have isPrivate set to true', () => {
		const fields = SETTINGS_CHAT_SECTIONS.flatMap((section) => section.fields);
		const apiKeyField = fields.find((field) => field?.key === SETTINGS_KEYS.API_KEY);

		expect(apiKeyField).toBeDefined();
		expect(apiKeyField?.isPrivate).toBe(true);
	});
});
