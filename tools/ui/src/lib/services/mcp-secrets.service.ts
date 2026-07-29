/**
 * Stores MCP server auth headers outside of the plaintext settings blob.
 *
 * The map of serverId -> headers is persisted under a dedicated localStorage
 * key: as plaintext JSON when encryption is disabled, or as an `enc1:`
 * ciphertext value when it is enabled (see EncryptionService). Reads are
 * served synchronously from an in-memory cache that {@link load} populates;
 * while encryption is locked the cache stays empty.
 */

import { EncryptionService } from '$lib/services/encryption.service';
import { MCP_SECRETS_LOCALSTORAGE_KEY } from '$lib/constants';

export class McpSecretsService {
	private static cache: Record<string, string> = {};

	/**
	 * Loads the stored map into the in-memory cache. A ciphertext value is
	 * only decrypted when the session is unlocked; otherwise the cache stays
	 * empty until the next load.
	 */
	static async load(): Promise<void> {
		this.cache = {};

		const raw = localStorage.getItem(MCP_SECRETS_LOCALSTORAGE_KEY);
		if (raw) {
			try {
				if (EncryptionService.isEncryptedValue(raw)) {
					if (EncryptionService.isUnlocked()) {
						this.cache = await EncryptionService.decryptJson<Record<string, string>>(raw);
					}
				} else {
					this.cache = JSON.parse(raw) as Record<string, string>;
				}
			} catch (error) {
				console.warn('[MCP] Failed to load stored server headers:', error);
			}
		}
	}

	/** The stored headers for a server, or undefined when absent or locked */
	static getHeaders(serverId: string): string | undefined {
		return this.cache[serverId] || undefined;
	}

	/**
	 * Sets (or clears, with undefined) a server's headers and persists.
	 *
	 * @throws Error if encryption is enabled but locked
	 */
	static async setHeaders(serverId: string, headers: string | undefined): Promise<void> {
		if (headers) {
			this.cache[serverId] = headers;
		} else {
			delete this.cache[serverId];
		}

		await this.persist();
	}

	/**
	 * Persists the cache with the current encryption state. Used by the
	 * encryption enable/disable flows to re-encrypt or decrypt the stored map.
	 *
	 * @param options.plaintext - Force a plaintext write (encryption disable flow)
	 * @throws Error if encryption is enabled but locked
	 */
	static async persist(options?: { plaintext?: boolean }): Promise<void> {
		if (Object.keys(this.cache).length === 0) {
			localStorage.removeItem(MCP_SECRETS_LOCALSTORAGE_KEY);
			return;
		}

		const json = JSON.stringify(this.cache);
		if (!options?.plaintext && EncryptionService.isEnabled()) {
			if (!EncryptionService.isUnlocked()) {
				throw new Error('Encryption is locked');
			}
			localStorage.setItem(
				MCP_SECRETS_LOCALSTORAGE_KEY,
				await EncryptionService.encryptString(json)
			);
		} else {
			localStorage.setItem(MCP_SECRETS_LOCALSTORAGE_KEY, json);
		}
	}
}
