import { describe, expect, it } from 'vitest';
import { isValidModelName, normalizeModelName } from '$lib/utils/model-names';

describe('normalizeModelName', () => {
	it('preserves Hugging Face org/model format (single slash)', () => {
		// Single slash is treated as Hugging Face format and preserved
		expect(normalizeModelName('meta-llama/Llama-3.1-8B')).toBe('meta-llama/Llama-3.1-8B');
		expect(normalizeModelName('models/model-name-1')).toBe('models/model-name-1');
	});

	it('extracts filename from multi-segment paths', () => {
		// Multiple slashes -> extract just the filename
		expect(normalizeModelName('path/to/model/model-name-2')).toBe('model-name-2');
		expect(normalizeModelName('/absolute/path/to/model')).toBe('model');
	});

	it('extracts filename from backslash paths', () => {
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
