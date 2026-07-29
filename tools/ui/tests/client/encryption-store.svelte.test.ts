import { afterEach, describe, expect, it, vi } from 'vitest';
import { encryptionStore } from '$lib/stores/encryption.svelte';
import { conversationsStore } from '$lib/stores/conversations.svelte';
import { DatabaseService } from '$lib/services/database.service';
import { EncryptionService } from '$lib/services/encryption.service';
import {
	ENCRYPTION_META_LOCALSTORAGE_KEY,
	ENCRYPTION_IDLE_TIMEOUT_LOCALSTORAGE_KEY,
	ENCRYPTION_SESSION_LOCALSTORAGE_KEY
} from '$lib/constants';

afterEach(() => {
	encryptionStore.disable();
	localStorage.removeItem(ENCRYPTION_META_LOCALSTORAGE_KEY);
	localStorage.removeItem(ENCRYPTION_IDLE_TIMEOUT_LOCALSTORAGE_KEY);
	localStorage.removeItem(ENCRYPTION_SESSION_LOCALSTORAGE_KEY);
	vi.useRealTimers();
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

	it('releases every concurrent ensureUnlocked caller on unlock', async () => {
		await encryptionStore.setupWithPassphrase('pw');
		encryptionStore.lock();

		let first = false;
		let second = false;
		const gates = Promise.all([
			encryptionStore.ensureUnlocked().then(() => {
				first = true;
			}),
			encryptionStore.ensureUnlocked().then(() => {
				second = true;
			})
		]);

		expect(await encryptionStore.unlockWithPassphrase('pw')).toBe(true);
		await gates;
		expect(first).toBe(true);
		expect(second).toBe(true);

		// a subsequent lock starts a fresh gate
		encryptionStore.lock();
		expect(encryptionStore.needsUnlock).toBe(true);
		let third = false;
		const gate = encryptionStore.ensureUnlocked().then(() => {
			third = true;
		});
		await Promise.resolve();
		expect(third).toBe(false);
		expect(await encryptionStore.unlockWithPassphrase('pw')).toBe(true);
		await gate;
		expect(third).toBe(true);
	});
});

describe('encryptionStore idle timer', () => {
	it('auto-locks after the configured idle timeout', async () => {
		vi.useFakeTimers();

		// Set a 1-minute idle timeout
		localStorage.setItem(ENCRYPTION_IDLE_TIMEOUT_LOCALSTORAGE_KEY, '1');
		encryptionStore.idleTimeoutMinutes = 1;

		await encryptionStore.setupWithPassphrase('pw');
		expect(encryptionStore.isUnlocked).toBe(true);

		// Fast-forward 1 minute
		vi.advanceTimersByTime(60_000);
		await Promise.resolve(); // let the setTimeout callback fire

		expect(encryptionStore.isUnlocked).toBe(false);
		expect(encryptionStore.needsUnlock).toBe(true);

		vi.useRealTimers();
	});

	it('does not auto-lock when idle timeout is 0 (Never)', async () => {
		vi.useFakeTimers();

		localStorage.setItem(ENCRYPTION_IDLE_TIMEOUT_LOCALSTORAGE_KEY, '0');
		encryptionStore.idleTimeoutMinutes = 0;

		await encryptionStore.setupWithPassphrase('pw');
		expect(encryptionStore.isUnlocked).toBe(true);

		vi.advanceTimersByTime(60_000);
		await Promise.resolve();

		expect(encryptionStore.isUnlocked).toBe(true);

		vi.useRealTimers();
	});

	it('resets the idle timer on user activity', async () => {
		vi.useFakeTimers();

		localStorage.setItem(ENCRYPTION_IDLE_TIMEOUT_LOCALSTORAGE_KEY, '1');
		encryptionStore.idleTimeoutMinutes = 1;

		await encryptionStore.setupWithPassphrase('pw');
		expect(encryptionStore.isUnlocked).toBe(true);

		// Advance 55 seconds, then simulate user activity (resets timer)
		vi.advanceTimersByTime(55_000);
		encryptionStore.onUserActivity();
		await Promise.resolve();

		// Advance another 55 seconds (total 110s from start, but only 55s since reset)
		vi.advanceTimersByTime(55_000);
		await Promise.resolve();

		// Should still be unlocked (55s < 60s timeout)
		expect(encryptionStore.isUnlocked).toBe(true);

		// Advance 5 more seconds to cross the 60s threshold from the reset
		vi.advanceTimersByTime(5_000);
		await Promise.resolve();

		expect(encryptionStore.isUnlocked).toBe(false);

		vi.useRealTimers();
	});

	it('clears the idle timer on lock', async () => {
		vi.useFakeTimers();

		localStorage.setItem(ENCRYPTION_IDLE_TIMEOUT_LOCALSTORAGE_KEY, '1');
		encryptionStore.idleTimeoutMinutes = 1;

		await encryptionStore.setupWithPassphrase('pw');
		expect(encryptionStore.isUnlocked).toBe(true);

		// Advance 55 seconds
		vi.advanceTimersByTime(55_000);

		// Manually lock — timer should be cleared
		encryptionStore.lock();
		expect(encryptionStore.isUnlocked).toBe(false);

		// Advance past where the old timer would have fired
		vi.advanceTimersByTime(10_000);
		await Promise.resolve();

		// Should still be locked (no re-lock from stale timer), and needsUnlock should be stable
		expect(encryptionStore.isUnlocked).toBe(false);
		expect(encryptionStore.needsUnlock).toBe(true);

		vi.useRealTimers();
	});

	it('exposes the idle deadline while unlocked and clears it on lock', async () => {
		vi.useFakeTimers();

		encryptionStore.idleTimeoutMinutes = 5;
		await encryptionStore.setupWithPassphrase('pw');

		expect(encryptionStore.idleDeadlineAt).toBe(Date.now() + 5 * 60_000);

		encryptionStore.onUserActivity();
		expect(encryptionStore.idleDeadlineAt).toBe(Date.now() + 5 * 60_000);

		encryptionStore.lock();
		expect(encryptionStore.idleDeadlineAt).toBeNull();

		vi.useRealTimers();
	});

	it('has no idle deadline when idle timeout is Never', async () => {
		encryptionStore.idleTimeoutMinutes = 0;
		await encryptionStore.setupWithPassphrase('pw');

		expect(encryptionStore.isUnlocked).toBe(true);
		expect(encryptionStore.idleDeadlineAt).toBeNull();
	});

	it('notifies lock/unlock listeners and supports unsubscribe', async () => {
		const events: string[] = [];
		const offLock = encryptionStore.onLock(() => events.push('lock'));
		const offUnlock = encryptionStore.onUnlock(() => events.push('unlock'));

		await encryptionStore.setupWithPassphrase('pw');
		expect(events).toEqual([]);

		encryptionStore.lock();
		expect(events).toEqual(['lock']);

		expect(await encryptionStore.unlockWithPassphrase('pw')).toBe(true);
		expect(events).toEqual(['lock', 'unlock']);

		offLock();
		offUnlock();
		encryptionStore.lock();
		expect(events).toEqual(['lock', 'unlock']);
	});

	it('persists and reloads idle timeout preference', () => {
		// Set to 15 minutes
		encryptionStore.setIdleTimeout(15);
		expect(encryptionStore.idleTimeoutMinutes).toBe(15);
		expect(localStorage.getItem(ENCRYPTION_IDLE_TIMEOUT_LOCALSTORAGE_KEY)).toBe('15');

		// Create a fresh store-like read to verify persistence
		const stored = localStorage.getItem(ENCRYPTION_IDLE_TIMEOUT_LOCALSTORAGE_KEY);
		expect(stored).toBe('15');

		// Change to "Never"
		encryptionStore.setIdleTimeout(0);
		expect(encryptionStore.idleTimeoutMinutes).toBe(0);
		expect(localStorage.getItem(ENCRYPTION_IDLE_TIMEOUT_LOCALSTORAGE_KEY)).toBe('0');
	});
});

describe('encryptionStore session resume', () => {
	it('resumes the session within the idle window without a passphrase', async () => {
		encryptionStore.idleTimeoutMinutes = 5;

		await encryptionStore.setupWithPassphrase('pw');
		const ciphertext = await EncryptionService.encryptString('secret');

		// simulate a page refresh: the in-memory key is gone, the record stays
		EncryptionService.lock();
		encryptionStore.refresh();
		expect(encryptionStore.isUnlocked).toBe(false);

		await encryptionStore.ensureUnlocked();

		expect(encryptionStore.isUnlocked).toBe(true);
		expect(encryptionStore.needsUnlock).toBe(false);
		expect(await EncryptionService.decryptString(ciphertext)).toBe('secret');
	});

	it('does not resume past the idle window and wipes the record', async () => {
		encryptionStore.idleTimeoutMinutes = 5;

		await encryptionStore.setupWithPassphrase('pw');

		// age the record beyond the window
		const record = JSON.parse(localStorage.getItem(ENCRYPTION_SESSION_LOCALSTORAGE_KEY)!);
		record.lastActivity = Date.now() - 10 * 60_000;
		localStorage.setItem(ENCRYPTION_SESSION_LOCALSTORAGE_KEY, JSON.stringify(record));

		EncryptionService.lock();
		encryptionStore.refresh();

		let released = false;
		const gate = encryptionStore.ensureUnlocked().then(() => {
			released = true;
		});
		await Promise.resolve();
		await Promise.resolve();

		expect(released).toBe(false);
		expect(encryptionStore.isUnlocked).toBe(false);
		expect(localStorage.getItem(ENCRYPTION_SESSION_LOCALSTORAGE_KEY)).toBeNull();

		expect(await encryptionStore.unlockWithPassphrase('pw')).toBe(true);
		await gate;
	});

	it('resumes indefinitely when idle timeout is Never', async () => {
		encryptionStore.idleTimeoutMinutes = 0;

		await encryptionStore.setupWithPassphrase('pw');

		const record = JSON.parse(localStorage.getItem(ENCRYPTION_SESSION_LOCALSTORAGE_KEY)!);
		record.lastActivity = Date.now() - 30 * 24 * 60 * 60_000;
		localStorage.setItem(ENCRYPTION_SESSION_LOCALSTORAGE_KEY, JSON.stringify(record));

		EncryptionService.lock();
		encryptionStore.refresh();

		await encryptionStore.ensureUnlocked();
		expect(encryptionStore.isUnlocked).toBe(true);
	});

	it('clears the resumable session on lock and disable', async () => {
		await encryptionStore.setupWithPassphrase('pw');
		expect(localStorage.getItem(ENCRYPTION_SESSION_LOCALSTORAGE_KEY)).not.toBeNull();

		encryptionStore.lock();
		expect(localStorage.getItem(ENCRYPTION_SESSION_LOCALSTORAGE_KEY)).toBeNull();

		expect(await encryptionStore.unlockWithPassphrase('pw')).toBe(true);
		expect(localStorage.getItem(ENCRYPTION_SESSION_LOCALSTORAGE_KEY)).not.toBeNull();

		encryptionStore.disable();
		expect(localStorage.getItem(ENCRYPTION_SESSION_LOCALSTORAGE_KEY)).toBeNull();
	});

	it('bumps lastActivity on user activity, throttled', async () => {
		vi.useFakeTimers();

		encryptionStore.idleTimeoutMinutes = 5;
		await encryptionStore.setupWithPassphrase('pw');

		const readLastActivity = () =>
			JSON.parse(localStorage.getItem(ENCRYPTION_SESSION_LOCALSTORAGE_KEY)!).lastActivity;
		const initial = readLastActivity();

		vi.advanceTimersByTime(10_000);
		encryptionStore.onUserActivity();
		expect(readLastActivity()).toBe(initial);

		vi.advanceTimersByTime(31_000);
		encryptionStore.onUserActivity();
		await vi.waitFor(() => {
			expect(readLastActivity()).toBeGreaterThan(initial);
		});

		vi.useRealTimers();
	});
});

describe('lock data purge', () => {
	afterEach(async () => {
		conversationsStore.purgeDecryptedData();
		const convs = await DatabaseService.getAllConversations();
		await DatabaseService.bulkDeleteConversations(convs.map((c) => c.id));
	});

	it('purgeDecryptedData drops decrypted state and resets init', async () => {
		await encryptionStore.setupWithPassphrase('pw');

		const conv = await DatabaseService.createConversation('secret chat');
		await conversationsStore.loadConversations();
		expect(await conversationsStore.loadConversation(conv.id)).toBe(true);
		conversationsStore.isInitialized = true;

		expect(conversationsStore.conversations.some((c) => c.name === 'secret chat')).toBe(true);
		expect(conversationsStore.activeConversation?.name).toBe('secret chat');

		conversationsStore.purgeDecryptedData();

		expect(conversationsStore.conversations).toEqual([]);
		expect(conversationsStore.activeConversation).toBeNull();
		expect(conversationsStore.activeMessages).toEqual([]);
		expect(conversationsStore.isInitialized).toBe(false);
	});

	it('reloads decrypted data via init after a lock/unlock cycle', async () => {
		await encryptionStore.setupWithPassphrase('pw');
		await DatabaseService.createConversation('secret chat');
		await conversationsStore.loadConversations();

		conversationsStore.purgeDecryptedData();
		encryptionStore.lock();
		expect(conversationsStore.conversations).toEqual([]);

		expect(await encryptionStore.unlockWithPassphrase('pw')).toBe(true);

		await conversationsStore.init();
		expect(conversationsStore.isInitialized).toBe(true);
		expect(conversationsStore.conversations.some((c) => c.name === 'secret chat')).toBe(true);
	});
});
