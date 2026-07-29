import { afterEach, describe, expect, it } from 'vitest';
import { EncryptionService } from '$lib/services/encryption.service';
import { ENCRYPTION_META_LOCALSTORAGE_KEY } from '$lib/constants';

afterEach(() => {
	EncryptionService.lock();
	localStorage.removeItem(ENCRYPTION_META_LOCALSTORAGE_KEY);
});

describe('EncryptionService', () => {
	it('is supported in this context and starts disabled and locked', () => {
		expect(EncryptionService.isSupported()).toBe(true);
		expect(EncryptionService.isEnabled()).toBe(false);
		expect(EncryptionService.isUnlocked()).toBe(false);
	});

	it('sets up with a passphrase and unlocks the session', async () => {
		await EncryptionService.setupWithPassphrase('correct horse battery staple');

		expect(EncryptionService.isEnabled()).toBe(true);
		expect(EncryptionService.isUnlocked()).toBe(true);
	});

	it('round-trips strings and uses a random IV per call', async () => {
		await EncryptionService.setupWithPassphrase('pw');

		const plaintext = 'hello world\nwith newlines\tand tabs';
		const first = await EncryptionService.encryptString(plaintext);
		const second = await EncryptionService.encryptString(plaintext);

		expect(first).not.toBe(second);
		expect(EncryptionService.isEncryptedValue(first)).toBe(true);
		expect(EncryptionService.isEncryptedValue(plaintext)).toBe(false);
		expect(await EncryptionService.decryptString(first)).toBe(plaintext);
		expect(await EncryptionService.decryptString(second)).toBe(plaintext);
	});

	it('rejects a wrong passphrase without unlocking', async () => {
		await EncryptionService.setupWithPassphrase('right');
		EncryptionService.lock();

		expect(await EncryptionService.unlockWithPassphrase('wrong')).toBe(false);
		expect(EncryptionService.isUnlocked()).toBe(false);

		expect(await EncryptionService.unlockWithPassphrase('right')).toBe(true);
		expect(EncryptionService.isUnlocked()).toBe(true);
	});

	it('refuses to encrypt or decrypt while locked', async () => {
		await EncryptionService.setupWithPassphrase('pw');
		EncryptionService.lock();

		await expect(EncryptionService.encryptString('x')).rejects.toThrow('Encryption is locked');
		await expect(EncryptionService.decryptString('enc1:aa.bb')).rejects.toThrow(
			'Encryption is locked'
		);
	});

	it('re-wraps the DEK on passphrase change and keeps data readable', async () => {
		await EncryptionService.setupWithPassphrase('old');
		const payload = await EncryptionService.encryptString('secret');

		expect(await EncryptionService.changePassphrase('nope', 'new')).toBe(false);
		expect(await EncryptionService.changePassphrase('old', 'new')).toBe(true);

		EncryptionService.lock();
		expect(await EncryptionService.unlockWithPassphrase('old')).toBe(false);
		expect(await EncryptionService.unlockWithPassphrase('new')).toBe(true);
		expect(await EncryptionService.decryptString(payload)).toBe('secret');
	});

	it('refuses to set up twice', async () => {
		await EncryptionService.setupWithPassphrase('pw');

		await expect(EncryptionService.setupWithPassphrase('pw2')).rejects.toThrow(
			'Encryption is already set up'
		);
	});

	it('round-trips JSON payloads, including large ones', async () => {
		await EncryptionService.setupWithPassphrase('pw');

		const extra = [{ type: 'text', name: 'a.txt', content: 'x'.repeat(100_000) }];
		const payload = await EncryptionService.encryptJson(extra);

		expect(await EncryptionService.decryptJson(payload)).toEqual(extra);
	});

	it('disable() clears the persisted meta and locks the session', async () => {
		await EncryptionService.setupWithPassphrase('pw');

		EncryptionService.disable();

		expect(EncryptionService.isEnabled()).toBe(false);
		expect(EncryptionService.isUnlocked()).toBe(false);
		expect(await EncryptionService.unlockWithPassphrase('pw')).toBe(false);
	});

	describe('meta validation', () => {
		it('accepts the metadata it persists', async () => {
			await EncryptionService.setupWithPassphrase('pw');

			const meta = EncryptionService.getPersistedMeta();
			expect(meta).not.toBeNull();
			expect(EncryptionService.isValidMeta(meta)).toBe(true);
		});

		it.each([
			['too few iterations', { kdfIterations: 1_000 }],
			['too many iterations', { kdfIterations: 2 ** 31 - 1 }],
			['non-integer iterations', { kdfIterations: 600_000.5 }],
			['string iterations', { kdfIterations: '600000' }],
			['wrong salt length', { salt: btoa('short') }],
			['malformed base64 iv', { wrapIv: '!!!not-base64!!!' }],
			['wrong wrapped dek length', { wrappedDek: btoa('too-short-for-a-wrapped-key') }],
			['wrong version', { version: 2 }]
		])('rejects %s', async (_label, override) => {
			await EncryptionService.setupWithPassphrase('pw');
			const meta = { ...EncryptionService.getPersistedMeta(), ...override } as EncryptionMeta;

			expect(EncryptionService.isValidMeta(meta)).toBe(false);
			// and the guard holds before any key derivation happens
			expect(await EncryptionService.unwrapDekWithPassphrase(meta, 'pw')).toBeNull();
		});

		it('treats tampered persisted metadata as not configured', async () => {
			await EncryptionService.setupWithPassphrase('pw');
			const meta = EncryptionService.getPersistedMeta()!;
			localStorage.setItem(
				ENCRYPTION_META_LOCALSTORAGE_KEY,
				JSON.stringify({ ...meta, kdfIterations: 2 ** 31 - 1 })
			);

			expect(EncryptionService.isEnabled()).toBe(false);
			expect(EncryptionService.getPersistedMeta()).toBeNull();
		});
	});
});
