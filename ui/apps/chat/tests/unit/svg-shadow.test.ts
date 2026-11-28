import { readSvgDimension } from '#lib/utils/svg-shadow.ts';
import { describe, expect, it } from 'vitest';

describe('svg-shadow readSvgDimension', () => {
	it('reads unitless and px lengths as user units', () => {
		expect(readSvgDimension('800')).toBe(800);
		expect(readSvgDimension('800px')).toBe(800);
		expect(readSvgDimension(' 12.5 ')).toBe(12.5);
	});

	it('converts absolute units at 96 dpi', () => {
		expect(readSvgDimension('1in')).toBe(96);
		expect(readSvgDimension('72pt')).toBeCloseTo(96, 10);
		expect(readSvgDimension('6pc')).toBe(96);
		expect(readSvgDimension('2.54cm')).toBeCloseTo(96, 10);
		expect(readSvgDimension('25.4mm')).toBeCloseTo(96, 10);
	});

	it('returns null for relative, missing, zero or malformed lengths', () => {
		expect(readSvgDimension(null)).toBeNull();
		expect(readSvgDimension('100%')).toBeNull();
		expect(readSvgDimension('10em')).toBeNull();
		expect(readSvgDimension('auto')).toBeNull();
		expect(readSvgDimension('0')).toBeNull();
		expect(readSvgDimension('')).toBeNull();
	});
});
