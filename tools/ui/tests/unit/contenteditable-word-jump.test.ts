import { describe, expect, it } from 'vitest';
import { badgeAwareWordJump, leadingBadgeEdgeOffset } from '$lib/utils';

// Layout of `hello [docs](file:///a/b) world foo`:
//   "hello" 0-4, " " 5, badge 6-24 (length 19), " " 25, "world" 26-30, " " 31, "foo" 32-34
const BADGE = '[docs](file:///a/b)';
const SOURCE = `hello ${BADGE} world foo`;
const BADGE_START = 6;
const BADGE_END = 25;

describe('badgeAwareWordJump', () => {
	it('returns null when the buffer has no badge', () => {
		expect(badgeAwareWordJump('hello world', 0, 'forward')).toBeNull();
		expect(badgeAwareWordJump('hello world', 11, 'backward')).toBeNull();
	});

	it('jumps forward onto a badge landing at its end, not the next word', () => {
		expect(badgeAwareWordJump(SOURCE, BADGE_START, 'forward')).toBe(BADGE_END);
	});

	it('jumps forward from the space before a badge landing at its end', () => {
		expect(badgeAwareWordJump(SOURCE, BADGE_START - 1, 'forward')).toBe(BADGE_END);
	});

	it('jumps backward over a badge landing at its start', () => {
		expect(badgeAwareWordJump(SOURCE, BADGE_END, 'backward')).toBe(BADGE_START);
	});

	it('jumps backward from the next word onto the badge start', () => {
		// caret at the start of "world"
		expect(badgeAwareWordJump(SOURCE, BADGE_END + 1, 'backward')).toBe(BADGE_START);
	});

	it('returns null for jumps that cross no badge', () => {
		// forward over "hello" only
		expect(badgeAwareWordJump(SOURCE, 0, 'forward')).toBeNull();
		// backward over "foo" only
		expect(badgeAwareWordJump(SOURCE, SOURCE.length, 'backward')).toBeNull();
		// backward away from the badge (over "hello")
		expect(badgeAwareWordJump(SOURCE, BADGE_START, 'backward')).toBeNull();
	});

	it('treats a leading badge as one word in both directions', () => {
		const source = `${BADGE} rest`;
		expect(badgeAwareWordJump(source, 0, 'forward')).toBe(BADGE.length);
		expect(badgeAwareWordJump(source, BADGE.length, 'backward')).toBe(0);
	});

	it('treats adjacent badges as separate words', () => {
		// each badge is 14 chars: "[a](file:///x)" / "[b](file:///y)"
		const source = '[a](file:///x)[b](file:///y)';
		expect(badgeAwareWordJump(source, 0, 'forward')).toBe(14);
		expect(badgeAwareWordJump(source, 14, 'forward')).toBe(28);
		expect(badgeAwareWordJump(source, 28, 'backward')).toBe(14);
		expect(badgeAwareWordJump(source, 14, 'backward')).toBe(0);
	});

	it('jumps over a badge following punctuation', () => {
		// "foo," 0-3, " " 4, badge 5-23 (end 24), " bar" 24-27
		const source = `foo, ${BADGE} bar`;
		expect(badgeAwareWordJump(source, 0, 'forward')).toBeNull();
		expect(badgeAwareWordJump(source, 3, 'forward')).toBe(24);
	});
});

describe('leadingBadgeEdgeOffset', () => {
	it('returns 0 when the caret sits exactly at a leading badge end', () => {
		expect(leadingBadgeEdgeOffset(`${BADGE} rest`, BADGE.length)).toBe(0);
	});

	it('returns null when the caret is anywhere else', () => {
		expect(leadingBadgeEdgeOffset(`${BADGE} rest`, 0)).toBeNull();
		expect(leadingBadgeEdgeOffset(`${BADGE} rest`, BADGE.length + 2)).toBeNull();
	});

	it('returns null when the buffer does not start with a badge', () => {
		expect(leadingBadgeEdgeOffset(SOURCE, BADGE_END)).toBeNull();
		expect(leadingBadgeEdgeOffset('', 0)).toBeNull();
	});
});
