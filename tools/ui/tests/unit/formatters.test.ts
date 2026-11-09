import {
	MS_PER_SECOND,
	RELATIVE_TIME_MAX_DAYS,
	SECONDS_PER_DAY,
	SECONDS_PER_HOUR,
	SECONDS_PER_MINUTE
} from '$lib/constants';
import { formatContextLength, formatParameters, formatRelativeTime } from '$lib/utils/formatters';
import { afterEach, beforeEach, describe, expect, it, vi } from 'vitest';

describe('formatContextLength', () => {
	it('shows a dash when no value is known', () => {
		expect(formatContextLength([])).toBe('-');
	});

	it('prints the value with thousands separators', () => {
		expect(formatContextLength([8192])).toBe('8,192 tokens');
	});

	it('pairs the configured context with the supported one', () => {
		expect(formatContextLength([8192, 131072])).toBe('8,192 / 131,072 tokens');
	});
});

describe('formatParameters', () => {
	it('shortens large parameter counts', () => {
		expect(formatParameters(35_000_000_000)).toBe('35B');
		expect(formatParameters(3_000_000)).toBe('3.00M');
		expect(formatParameters(512)).toBe('512');
	});
});

describe('formatRelativeTime', () => {
	const NOW = new Date('2026-06-15T12:00:00Z').getTime();
	const ago = (seconds: number) => NOW - seconds * MS_PER_SECOND;

	beforeEach(() => {
		vi.useFakeTimers();
		vi.setSystemTime(NOW);
	});

	afterEach(() => {
		vi.useRealTimers();
	});

	it('says just now under a minute', () => {
		expect(formatRelativeTime(ago(SECONDS_PER_MINUTE - 1))).toBe('just now');
	});

	it('counts minutes, hours and days with singular and plural units', () => {
		expect(formatRelativeTime(ago(SECONDS_PER_MINUTE))).toBe('1 minute ago');
		expect(formatRelativeTime(ago(37 * SECONDS_PER_MINUTE))).toBe('37 minutes ago');
		expect(formatRelativeTime(ago(SECONDS_PER_HOUR))).toBe('1 hour ago');
		expect(formatRelativeTime(ago(5 * SECONDS_PER_HOUR))).toBe('5 hours ago');
		expect(formatRelativeTime(ago(SECONDS_PER_DAY))).toBe('1 day ago');
		expect(formatRelativeTime(ago(3 * SECONDS_PER_DAY))).toBe('3 days ago');
	});

	it('falls back to a date once the age reaches the day limit', () => {
		const timestamp = ago(RELATIVE_TIME_MAX_DAYS * SECONDS_PER_DAY);

		expect(formatRelativeTime(timestamp)).toBe(new Date(timestamp).toLocaleDateString());
	});
});
