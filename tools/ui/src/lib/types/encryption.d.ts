/**
 * Persisted parameters for at-rest encryption: the data encryption key (DEK)
 * wrapped with a passphrase-derived key encryption key (KEK), plus the KDF
 * parameters needed to re-derive that KEK. Stored in localStorage; on its own
 * it contains no usable key material - the passphrase never leaves memory.
 */
export interface EncryptionMeta {
	/** Format version, currently 1 */
	version: number;
	kdf: 'PBKDF2';
	kdfHash: 'SHA-256';
	kdfIterations: number;
	/** Base64-encoded PBKDF2 salt */
	salt: string;
	/** Base64-encoded AES-GCM IV used when wrapping the DEK */
	wrapIv: string;
	/** Base64-encoded DEK wrapped with the KEK */
	wrappedDek: string;
}
