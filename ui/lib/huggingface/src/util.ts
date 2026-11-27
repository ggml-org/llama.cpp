/** Small helpers the package's modules share. */

import { HF_BASE64_CHUNK_BYTES, HF_REPO_ID_SEPARATOR } from './constants.js';

/** The text of anything thrown: an Error's message, else the value as a string. */
export function errorMessage(error: unknown): string {
	return error instanceof Error ? error.message : String(error);
}

/** Base64 of bytes, a chunk at a time, under the argument limit of String.fromCharCode. */
export function bytesToBase64(bytes: Uint8Array): string {
	let binary = '';

	for (let i = 0; i < bytes.length; i += HF_BASE64_CHUNK_BYTES) {
		binary += String.fromCharCode(...bytes.subarray(i, i + HF_BASE64_CHUNK_BYTES));
	}

	return btoa(binary);
}

/** Base64 for a URL: `-` and `_` for `+` and `/`, no padding (what PKCE and a state use). */
export function bytesToBase64Url(bytes: Uint8Array): string {
	return bytesToBase64(bytes).replaceAll('+', '-').replaceAll('/', '_').replaceAll('=', '');
}

/**
 * Splits the repo ID `owner/name` of a dataset into its owner and its name. An ID without a
 * separator is all name.
 */
export function splitRepoId(repoId: string): [owner: string, name: string] {
	const at = repoId.indexOf(HF_REPO_ID_SEPARATOR);

	return at === -1 ? ['', repoId] : [repoId.slice(0, at), repoId.slice(at + 1)];
}
