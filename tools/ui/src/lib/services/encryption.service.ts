/**
 * Encryption Service - at-rest encryption for IndexedDB data
 *
 * Envelope encryption: a random 256-bit AES-GCM data encryption key (DEK)
 * encrypts record fields. The DEK itself is wrapped by a key encryption key
 * (KEK) derived from the user's passphrase via PBKDF2; only the wrapped DEK
 * and the KDF parameters are persisted (localStorage). The passphrase never
 * leaves memory, so neither the server nor someone holding a copy of the
 * browser profile can read the data without it.
 *
 * Session model: the unwrapped DEK is held in memory only, as a
 * non-extractable CryptoKey. Locking (or closing the tab) drops it, and the
 * next session starts locked until the passphrase unlocks it again.
 */

import { ENCRYPTION_META_LOCALSTORAGE_KEY } from '$lib/constants';

const ENCRYPTION_FORMAT_VERSION = 1;
const ENCRYPTED_VALUE_PREFIX = 'enc1:';
const KDF_ITERATIONS = 600_000;
const AES_GCM_IV_BYTES = 12;
const PBKDF2_SALT_BYTES = 16;

function bytesToBase64(bytes: Uint8Array): string {
	const CHUNK = 0x8000;
	let binary = '';
	for (let i = 0; i < bytes.length; i += CHUNK) {
		binary += String.fromCharCode(...bytes.subarray(i, i + CHUNK));
	}
	return btoa(binary);
}

function base64ToBytes(base64: string): Uint8Array {
	const binary = atob(base64);
	const bytes = new Uint8Array(binary.length);
	for (let i = 0; i < binary.length; i++) {
		bytes[i] = binary.charCodeAt(i);
	}
	return bytes;
}

function deriveKek(passphrase: string, salt: Uint8Array, iterations: number): Promise<CryptoKey> {
	const encoder = new TextEncoder();
	return globalThis.crypto.subtle
		.importKey('raw', encoder.encode(passphrase), 'PBKDF2', false, ['deriveKey'])
		.then((material) =>
			globalThis.crypto.subtle.deriveKey(
				{ name: 'PBKDF2', hash: 'SHA-256', salt: salt as BufferSource, iterations },
				material,
				{ name: 'AES-GCM', length: 256 },
				false,
				['wrapKey', 'unwrapKey']
			)
		);
}

export class EncryptionService {
	private static dek: CryptoKey | null = null;

	/** WebCrypto requires a secure context (https or localhost) */
	static isSupported(): boolean {
		return typeof globalThis.crypto?.subtle !== 'undefined';
	}

	/** Encryption is enabled once a wrapped DEK has been persisted */
	static isEnabled(): boolean {
		return this.readMeta() !== null;
	}

	/** The session DEK is in memory */
	static isUnlocked(): boolean {
		return this.dek !== null;
	}

	/** Drops the session DEK; persisted data stays encrypted */
	static lock(): void {
		this.dek = null;
	}

	/**
	 * Disables encryption entirely: removes the wrapped DEK and locks the
	 * session. Callers must decrypt all data beforehand; anything left
	 * encrypted becomes unrecoverable.
	 */
	static disable(): void {
		localStorage.removeItem(ENCRYPTION_META_LOCALSTORAGE_KEY);
		this.lock();
	}

	/**
	 * Sets up encryption with a new passphrase: generates a fresh DEK, wraps
	 * it with a passphrase-derived KEK and persists only the wrapped blob.
	 *
	 * @param passphrase - The user's passphrase
	 * @throws Error if the passphrase is empty or encryption is already set up
	 */
	static async setupWithPassphrase(passphrase: string): Promise<void> {
		if (!passphrase) {
			throw new Error('Passphrase must not be empty');
		}
		if (this.readMeta() !== null) {
			throw new Error('Encryption is already set up');
		}

		const salt = globalThis.crypto.getRandomValues(new Uint8Array(PBKDF2_SALT_BYTES));
		const kek = await deriveKek(passphrase, salt, KDF_ITERATIONS);

		// extractable so it can be wrapped; the session copy below is not
		const dek = await globalThis.crypto.subtle.generateKey({ name: 'AES-GCM', length: 256 }, true, [
			'encrypt',
			'decrypt'
		]);
		const wrapIv = globalThis.crypto.getRandomValues(new Uint8Array(AES_GCM_IV_BYTES));
		const wrappedDek = await globalThis.crypto.subtle.wrapKey('raw', dek, kek, {
			name: 'AES-GCM',
			iv: wrapIv as BufferSource
		});

		this.writeMeta({
			version: ENCRYPTION_FORMAT_VERSION,
			kdf: 'PBKDF2',
			kdfHash: 'SHA-256',
			kdfIterations: KDF_ITERATIONS,
			salt: bytesToBase64(salt),
			wrapIv: bytesToBase64(wrapIv),
			wrappedDek: bytesToBase64(new Uint8Array(wrappedDek))
		});

		const rawDek = await globalThis.crypto.subtle.exportKey('raw', dek);
		this.dek = await globalThis.crypto.subtle.importKey(
			'raw',
			rawDek,
			{ name: 'AES-GCM', length: 256 },
			false,
			['encrypt', 'decrypt']
		);
	}

	/**
	 * Unlocks the session by unwrapping the persisted DEK with the passphrase.
	 *
	 * @param passphrase - The user's passphrase
	 * @returns True when the passphrase matched, false otherwise
	 */
	static async unlockWithPassphrase(passphrase: string): Promise<boolean> {
		const meta = this.readMeta();
		if (!meta || !passphrase) return false;

		const kek = await deriveKek(passphrase, base64ToBytes(meta.salt), meta.kdfIterations);
		try {
			this.dek = await globalThis.crypto.subtle.unwrapKey(
				'raw',
				base64ToBytes(meta.wrappedDek) as BufferSource,
				kek,
				{ name: 'AES-GCM', iv: base64ToBytes(meta.wrapIv) as BufferSource },
				{ name: 'AES-GCM', length: 256 },
				false,
				['encrypt', 'decrypt']
			);
			return true;
		} catch {
			// AES-GCM auth tag mismatch: wrong passphrase
			return false;
		}
	}

	/**
	 * Re-wraps the DEK under a new passphrase. Record data is untouched; only
	 * the persisted wrapped blob is rewritten.
	 *
	 * @param current - The current passphrase
	 * @param next - The new passphrase
	 * @returns True when the current passphrase matched and the change was applied
	 */
	static async changePassphrase(current: string, next: string): Promise<boolean> {
		const meta = this.readMeta();
		if (!meta || !next) return false;

		const currentKek = await deriveKek(current, base64ToBytes(meta.salt), meta.kdfIterations);
		let rawDek: ArrayBuffer;
		try {
			const tempDek = await globalThis.crypto.subtle.unwrapKey(
				'raw',
				base64ToBytes(meta.wrappedDek) as BufferSource,
				currentKek,
				{ name: 'AES-GCM', iv: base64ToBytes(meta.wrapIv) as BufferSource },
				{ name: 'AES-GCM', length: 256 },
				true,
				['encrypt', 'decrypt']
			);
			rawDek = await globalThis.crypto.subtle.exportKey('raw', tempDek);
		} catch {
			return false;
		}

		const salt = globalThis.crypto.getRandomValues(new Uint8Array(PBKDF2_SALT_BYTES));
		const nextKek = await deriveKek(next, salt, KDF_ITERATIONS);
		const wrapIv = globalThis.crypto.getRandomValues(new Uint8Array(AES_GCM_IV_BYTES));
		const nextDek = await globalThis.crypto.subtle.importKey(
			'raw',
			rawDek,
			{ name: 'AES-GCM', length: 256 },
			true,
			['encrypt', 'decrypt']
		);
		const wrappedDek = await globalThis.crypto.subtle.wrapKey('raw', nextDek, nextKek, {
			name: 'AES-GCM',
			iv: wrapIv as BufferSource
		});

		this.writeMeta({
			version: ENCRYPTION_FORMAT_VERSION,
			kdf: 'PBKDF2',
			kdfHash: 'SHA-256',
			kdfIterations: KDF_ITERATIONS,
			salt: bytesToBase64(salt),
			wrapIv: bytesToBase64(wrapIv),
			wrappedDek: bytesToBase64(new Uint8Array(wrappedDek))
		});

		this.dek = await globalThis.crypto.subtle.importKey(
			'raw',
			rawDek,
			{ name: 'AES-GCM', length: 256 },
			false,
			['encrypt', 'decrypt']
		);
		return true;
	}

	/** The persisted wrapped DEK + KDF parameters, or null when not set up */
	static getPersistedMeta(): EncryptionMeta | null {
		return this.readMeta();
	}

	/** Reports whether a value carries the encrypted-value prefix */
	static isEncryptedValue(value: unknown): value is string {
		return typeof value === 'string' && value.startsWith(ENCRYPTED_VALUE_PREFIX);
	}

	/**
	 * Encrypts a string with the session DEK, using a random IV per call.
	 *
	 * @returns `enc1:<base64 iv>.<base64 ciphertext>`
	 * @throws Error if the session is locked
	 */
	static async encryptString(plaintext: string): Promise<string> {
		if (!this.dek) throw new Error('Encryption is locked');

		const iv = globalThis.crypto.getRandomValues(new Uint8Array(AES_GCM_IV_BYTES));
		const ciphertext = await globalThis.crypto.subtle.encrypt(
			{ name: 'AES-GCM', iv: iv as BufferSource },
			this.dek,
			new TextEncoder().encode(plaintext)
		);
		return `${ENCRYPTED_VALUE_PREFIX}${bytesToBase64(iv)}.${bytesToBase64(new Uint8Array(ciphertext))}`;
	}

	/**
	 * Decrypts a value produced by {@link encryptString}.
	 *
	 * @throws Error if the session is locked or the value is not encrypted
	 */
	static async decryptString(payload: string): Promise<string> {
		if (!this.dek) throw new Error('Encryption is locked');
		if (!this.isEncryptedValue(payload)) throw new Error('Value is not encrypted');

		const body = payload.slice(ENCRYPTED_VALUE_PREFIX.length);
		const separator = body.indexOf('.');
		const iv = base64ToBytes(body.slice(0, separator));
		const ciphertext = base64ToBytes(body.slice(separator + 1));
		const plaintext = await globalThis.crypto.subtle.decrypt(
			{ name: 'AES-GCM', iv: iv as BufferSource },
			this.dek,
			ciphertext as BufferSource
		);
		return new TextDecoder().decode(plaintext);
	}

	/** Encrypts a JSON-serializable value; see {@link encryptString} */
	static async encryptJson(value: unknown): Promise<string> {
		return this.encryptString(JSON.stringify(value));
	}

	/** Decrypts a value produced by {@link encryptJson} and parses the JSON */
	static async decryptJson<T>(payload: string): Promise<T> {
		return JSON.parse(await this.decryptString(payload)) as T;
	}

	private static readMeta(): EncryptionMeta | null {
		try {
			const raw = localStorage.getItem(ENCRYPTION_META_LOCALSTORAGE_KEY);
			if (!raw) return null;
			const meta = JSON.parse(raw) as EncryptionMeta;
			return meta.version === ENCRYPTION_FORMAT_VERSION ? meta : null;
		} catch {
			return null;
		}
	}

	private static writeMeta(meta: EncryptionMeta): void {
		localStorage.setItem(ENCRYPTION_META_LOCALSTORAGE_KEY, JSON.stringify(meta));
	}
}
