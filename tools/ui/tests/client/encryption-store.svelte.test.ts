import { afterEach, describe, expect, it } from 'vitest';
import { encryptionStore } from '$lib/stores/encryption.svelte';
import { ENCRYPTION_META_LOCALSTORAGE_KEY } from '$lib/constants';

afterEach(() => {
	encryptionStore.disable();
	localStorage.removeItem(ENCRYPTION_META_LOCALSTORAGE_KEY);
});

describe('encryptionStore', () => {
	it('resolves ensureUnlocked immediately when encryption is disabled', async () => {
		encryptionStore.refresh();

		await encryptionStore.ensureUnlocked();
		expect(encryptionStore.needsUnlock).toBe(false);
	});

	it('blocks ensureUnlocked while locked and releases it on unlock', async () => {
		await encryptionStore.setupWithPassphrase('pw');
		encryptionStore.lock();

		expect(encryptionStore.needsUnlock).toBe(true);

		let released = false;
		const gate = encryptionStore.ensureUnlocked().then(() => {
			released = true;
		});

		// still pending: a microtask tick is not enough to release it
		await Promise.resolve();
		expect(released).toBe(false);

		expect(await encryptionStore.unlockWithPassphrase('wrong')).toBe(false);
		expect(encryptionStore.needsUnlock).toBe(true);
		await Promise.resolve();
		expect(released).toBe(false);

		expect(await encryptionStore.unlockWithPassphrase('pw')).toBe(true);
		await gate;
		expect(released).toBe(true);
		expect(encryptionStore.needsUnlock).toBe(false);
	});
});
