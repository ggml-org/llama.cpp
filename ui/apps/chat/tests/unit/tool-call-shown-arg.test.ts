import { SETTINGS_KEYS } from '#lib/constants/index.ts';
import { parseToolCallShownArgOverrides, toolCallShownArg } from '#lib/utils/index.ts';
import { describe, expect, it } from 'vitest';

function config(shownArg: string, overrides: Record<string, string> | string = {}) {
	return {
		[SETTINGS_KEYS.TOOL_CALL_SHOWN_ARG]: shownArg,
		[SETTINGS_KEYS.TOOL_CALL_SHOWN_ARG_OVERRIDES]:
			typeof overrides === 'string' ? overrides : JSON.stringify(overrides)
	};
}

describe('toolCallShownArg', () => {
	it('uses the global argument for a tool without its own', () => {
		expect(toolCallShownArg(config('description'), 'get_weather')).toBe('description');
	});

	it("prefers the tool's own argument", () => {
		expect(toolCallShownArg(config('description', { get_weather: 'city' }), 'get_weather')).toBe(
			'city'
		);
		expect(toolCallShownArg(config('description', { get_weather: 'city' }), 'other')).toBe(
			'description'
		);
	});

	it('returns null for none, globally or per tool', () => {
		expect(toolCallShownArg(config(''), 'get_weather')).toBeNull();
		expect(toolCallShownArg(config('description', { get_weather: '' }), 'get_weather')).toBeNull();
	});

	it('reads unreadable overrides as none', () => {
		expect(parseToolCallShownArgOverrides('not json')).toEqual({});
		expect(parseToolCallShownArgOverrides('[1]')).toEqual({});
		expect(parseToolCallShownArgOverrides('{"a":"x","b":1}')).toEqual({ a: 'x' });
		expect(toolCallShownArg(config('description', 'not json'), 'get_weather')).toBe('description');
	});
});
