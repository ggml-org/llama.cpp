/**
 * Key-value pair form constraints and sanitization patterns.
 *
 * Both regexes target characters dangerous in HTTP-header / env-var contexts:
 *   \x00        – null byte (injection)
 *   \x0A (\n)   – LF  (HTTP header injection / response splitting)
 *   \x0D (\r)   – CR  (HTTP header injection / response splitting)
 *   \x01–\x08, \x0B–\x0C, \x0E–\x1F, \x7F – other C0/DEL control chars
 *
 * KEY_UNSAFE_RE additionally strips TAB (\x09); values keep TAB because it is
 * a valid header-value continuation character per RFC 7230.
 */

export const KEY_VALUE_PAIR_KEY_MAX_LENGTH = 256;
export const KEY_VALUE_PAIR_VALUE_MAX_LENGTH = 8192;

// eslint-disable-next-line no-control-regex
export const KEY_VALUE_PAIR_UNSAFE_KEY_RE = /[\x00-\x1F\x7F]/g;
// eslint-disable-next-line no-control-regex
export const KEY_VALUE_PAIR_UNSAFE_VALUE_RE = /[\x00-\x08\x0A-\x0D\x0E-\x1F\x7F]/g;
