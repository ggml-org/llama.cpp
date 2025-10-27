import { describe, expect, it } from 'vitest';
import { isValidModelName, normalizeModelName } from './model-names';

describe('normalizeModelName', () => {
	it('extracts filename from forward slash path', () => {
		expect(normalizeModelName('models/model-name-1')).toBe('model-name-1');
		expect(normalizeModelName('path/to/model/model-name-2')).toBe('model-name-2');
	});

	it('extracts filename from backslash path', () => {
		expect(normalizeModelName('C\\Models\\model-name-1')).toBe('model-name-1');
		expect(normalizeModelName('path\\to\\model\\model-name-2')).toBe('model-name-2');
	});

	it('handles mixed path separators', () => {
		expect(normalizeModelName('path/to\\model/model-name-2')).toBe('model-name-2');
	});

	it('returns simple names as-is', () => {
		expect(normalizeModelName('simple-model')).toBe('simple-model');
		expect(normalizeModelName('model-name-2')).toBe('model-name-2');
	});

	it('trims whitespace', () => {
		expect(normalizeModelName('  model-name  ')).toBe('model-name');
	});

	it('returns empty string for empty input', () => {
		expect(normalizeModelName('')).toBe('');
		expect(normalizeModelName('   ')).toBe('');
	});
});

describe('isValidModelName', () => {
	it('returns true for valid names', () => {
		expect(isValidModelName('model')).toBe(true);
		expect(isValidModelName('path/to/model.bin')).toBe(true);
	});

	it('returns false for empty values', () => {
		expect(isValidModelName('')).toBe(false);
		expect(isValidModelName('   ')).toBe(false);
	});
});
