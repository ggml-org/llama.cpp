/**
 * 32-bit FNV-1a string hash.
 *
 * Deterministic, cheap, and sufficient as a *cache-equality* key - we only need
 * to detect "did the input change since last call", never cryptographic
 * properties. Choose hex output so the value is short, readable in logs, and
 * stable across re-implementations.
 */

const FNV_OFFSET_BASIS = 0x811c9dc5;
const FNV_PRIME = 0x01000193;
const FNV_RADIX = 16;

export function hashString(input: string): string {
	let hash = FNV_OFFSET_BASIS;
	for (let i = 0; i < input.length; i++) {
		hash ^= input.charCodeAt(i);
		hash = Math.imul(hash, FNV_PRIME);
	}
	return (hash >>> 0).toString(FNV_RADIX);
}
