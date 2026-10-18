import {
	formatPerfBandwidth,
	formatPerfGb,
	formatPerfMemory,
	formatPerfPercent,
	formatPerfRatio,
	formatPerfTps
} from '$lib/utils/perf-format';
import { describe, expect, it } from 'vitest';

const GIB = 1024 * 1024 * 1024;

describe('performance panel formats', () => {
	it('memory: base 1024, integral or >= 100 without decimals, else 1', () => {
		expect(formatPerfMemory(241 * GIB, 251 * GIB)).toBe('241 / 251 GB');
		expect(formatPerfMemory(11.2 * GIB, 12 * GIB)).toBe('11.2 / 12 GB');
		expect(formatPerfMemory(240.6 * GIB, 251.4 * GIB)).toBe('241 / 251 GB');
		expect(formatPerfMemory(11.96 * GIB, 12 * GIB)).toBe('12 / 12 GB');
		expect(formatPerfGb(0.5 * GIB)).toBe('0.5 GB');
		expect(formatPerfMemory(null, 12 * GIB)).toBe('n/a');
	});

	it('bandwidth: base 1000, GB/s from 1e9 up, else MB/s', () => {
		expect(formatPerfBandwidth(4.2e9)).toBe('4.2 GB/s');
		expect(formatPerfBandwidth(4.24e9)).toBe('4.2 GB/s');
		expect(formatPerfBandwidth(640e6)).toBe('640 MB/s');
		expect(formatPerfBandwidth(0)).toBe('0 MB/s');
		expect(formatPerfBandwidth(999.6e6)).toBe('1.0 GB/s');
		expect(formatPerfBandwidth(999.4e6)).toBe('999 MB/s');
		expect(formatPerfBandwidth(null)).toBe('n/a');
	});

	it('speed: 2 decimals below 10, else 1', () => {
		expect(formatPerfTps(0.53)).toBe('0.53 t/s');
		expect(formatPerfTps(0.525)).toBe('0.53 t/s');
		expect(formatPerfTps(14.8)).toBe('14.8 t/s');
		expect(formatPerfTps(9.996)).toBe('10.0 t/s');
		expect(formatPerfTps(9.994)).toBe('9.99 t/s');
		expect(formatPerfTps(1685.44)).toBe('1685.4 t/s');
		expect(formatPerfTps(undefined)).toBe('n/a');
	});

	it('percent: integer', () => {
		expect(formatPerfPercent(87)).toBe('87%');
		expect(formatPerfPercent(87.4)).toBe('87%');
		expect(formatPerfPercent(2.5)).toBe('3%');
		expect(formatPerfRatio(0.41)).toBe('41%');
		expect(formatPerfPercent(null)).toBe('n/a');
		expect(formatPerfPercent(NaN)).toBe('n/a');
	});
});
