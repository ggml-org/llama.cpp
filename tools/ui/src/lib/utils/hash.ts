/**
 * 32-bit FNV-1a string hash.
 *
 * Deterministic, cheap, and sufficient as a *cache-equality* key - we only need
 * to detect "did the input change since last call", never cryptographic
 * properties. Choose hex output so the value is short, readable in logs, and
 * stable across re-implementations.
 */
export function hashString(input: string): string {
	let hash = 0x811c9dc5;
	for (let i = 0; i < input.length; i++) {
		hash ^= input.charCodeAt(i);
		hash = Math.imul(hash, 0x01000193);
	}
	return (hash >>> 0).toString(16);
}
