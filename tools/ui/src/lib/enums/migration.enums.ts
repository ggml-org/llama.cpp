/**
 * Phase in the app lifecycle during which a migration runs. Boot migrations
 * run before the encryption unlock gate and must not read encrypted fields
 * (they would see ciphertext passthrough); post-unlock migrations run after
 * the gate and may assume the session is unlocked (or encryption is disabled).
 */
export enum MigrationPhase {
	BOOT = 'boot',
	POST_UNLOCK = 'post-unlock'
}
