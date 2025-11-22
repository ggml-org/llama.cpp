/**
 * Floating-point precision utilities
 *
 * Provides functions to normalize floating-point numbers for consistent comparison
 * and display, addressing JavaScript's floating-point precision issues.
 */

import { PRECISION_MULTIPLIER } from '$lib/constants/precision';

/**
 * Normalize floating-point numbers for consistent comparison
 * Addresses JavaScript floating-point precision issues (e.g., 0.949999988079071 → 0.95)
 */
export function normalizeFloatingPoint(value: unknown): unknown {
	return typeof value === 'number'
		? Math.round(value * PRECISION_MULTIPLIER) / PRECISION_MULTIPLIER
		: value;
}

/**
 * Type-safe version that only accepts numbers
 */
export function normalizeNumber(value: number): number {
	return Math.round(value * PRECISION_MULTIPLIER) / PRECISION_MULTIPLIER;
}
