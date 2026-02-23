import {
	KEY_VALUE_PAIR_KEY_MAX_LENGTH,
	KEY_VALUE_PAIR_VALUE_MAX_LENGTH,
	KEY_VALUE_PAIR_UNSAFE_KEY_RE,
	KEY_VALUE_PAIR_UNSAFE_VALUE_RE
} from '$lib/constants/key-value-pairs';

/**
 * Strip control characters unsafe in identifier/header-name contexts and cap length.
 * Removes all C0 controls (including TAB) and DEL.
 */
export function sanitizeKeyValuePairKey(raw: string): string {
	return raw.replace(KEY_VALUE_PAIR_UNSAFE_KEY_RE, '').slice(0, KEY_VALUE_PAIR_KEY_MAX_LENGTH);
}

/**
 * Strip control characters that enable header injection; allow TAB; cap length.
 * Removes null bytes, CR/LF and other C0/DEL controls while keeping TAB (\x09),
 * which is a valid header-value continuation character per RFC 7230.
 */
export function sanitizeKeyValuePairValue(raw: string): string {
	return raw.replace(KEY_VALUE_PAIR_UNSAFE_VALUE_RE, '').slice(0, KEY_VALUE_PAIR_VALUE_MAX_LENGTH);
}
