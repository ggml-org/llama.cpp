/**
 * Encryption-related constants (at-rest encryption for IndexedDB data).
 *
 * Centralized so the format stays shared between EncryptionService, the
 * stores and the database encryption seam.
 */

/** Format version of the persisted wrapped-DEK metadata */
export const ENCRYPTION_FORMAT_VERSION = 1;

/** Prefix marking a value as `enc1:<iv>.<ciphertext>` (see EncryptionService) */
export const ENCRYPTED_VALUE_PREFIX = 'enc1:';

/** PBKDF2 iterations used when (re)wrapping the DEK with a passphrase-derived KEK */
export const KDF_ITERATIONS = 600_000;

/** AES-GCM IV size in bytes */
export const AES_GCM_IV_BYTES = 12;

/** PBKDF2 salt size in bytes */
export const PBKDF2_SALT_BYTES = 16;

// 32-byte DEK + 16-byte GCM auth tag
/** Wrapped-DEK blob size in bytes */
export const WRAPPED_DEK_BYTES = 48;

// Accepted iteration band for persisted/imported metadata; guards against
// crafted imports freezing the tab on huge counts or weakening brute-force
// resistance with tiny ones
/** Minimum accepted PBKDF2 iteration count */
export const KDF_MIN_ITERATIONS = 100_000;

/** Maximum accepted PBKDF2 iteration count */
export const KDF_MAX_ITERATIONS = 10_000_000;

/** Key-derivation algorithm used for the KEK */
export const KEK_KDF = 'PBKDF2';

/** Hash used by PBKDF2 */
export const KEK_KDF_HASH = 'SHA-256';

/** Cipher algorithm used for both wrapping the DEK and encrypting record fields */
export const AES_GCM = 'AES-GCM';

/** Strict base64 matcher (no whitespace, 1-2 `=` padding) */
export const BASE64_REGEX = /^[A-Za-z0-9+/]*={0,2}$/;

/** Default idle timeout in minutes for auto-lock (0 = never) */
export const DEFAULT_IDLE_TIMEOUT_MINUTES = 5;

/** Selectable auto-lock idle timeouts in minutes */
export const IDLE_TIMEOUT_PRESETS = [
	{ value: 0, label: 'Never' },
	{ value: 1, label: '1 minute' },
	{ value: 5, label: '5 minutes' },
	{ value: 15, label: '15 minutes' },
	{ value: 30, label: '30 minutes' },
	{ value: 60, label: '1 hour' }
] as const;

/** Activity bumps rewrite the resumable session at most this often (ms) */
export const SESSION_WRITE_THROTTLE_MS = 30_000;
