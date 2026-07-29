/**
 * Encryption Store - reactive facade over EncryptionService
 *
 * Exposes the encryption session state (enabled/unlocked) as reactive state
 * and serializes the app bootstrap behind the unlock gate: `init()` awaits
 * `ensureUnlocked()`, which only resolves once the user has unlocked (or when
 * no unlock is needed), so no conversation data is ever loaded while locked.
 */

import { EncryptionService } from '$lib/services/encryption.service';

class EncryptionStore {
	isSupported = $state(false);
	isEnabled = $state(false);
	isUnlocked = $state(false);

	needsUnlock = $derived(this.isEnabled && !this.isUnlocked);

	private unlockResolver: (() => void) | null = null;

	refresh(): void {
		this.isSupported = EncryptionService.isSupported();
		this.isEnabled = EncryptionService.isEnabled();
		this.isUnlocked = EncryptionService.isUnlocked();
	}

	/**
	 * Resolves immediately unless encryption is enabled and locked; otherwise
	 * blocks until {@link unlockWithPassphrase} succeeds.
	 */
	ensureUnlocked(): Promise<void> {
		this.refresh();
		if (!this.isEnabled || this.isUnlocked) return Promise.resolve();

		return new Promise((resolve) => {
			this.unlockResolver = resolve;
		});
	}

	async unlockWithPassphrase(passphrase: string): Promise<boolean> {
		const unlocked = await EncryptionService.unlockWithPassphrase(passphrase);
		if (unlocked) {
			this.refresh();
			this.unlockResolver?.();
			this.unlockResolver = null;
		}
		return unlocked;
	}

	async setupWithPassphrase(passphrase: string): Promise<void> {
		await EncryptionService.setupWithPassphrase(passphrase);
		this.refresh();
	}

	async changePassphrase(current: string, next: string): Promise<boolean> {
		return await EncryptionService.changePassphrase(current, next);
	}

	lock(): void {
		EncryptionService.lock();
		this.refresh();
	}

	disable(): void {
		EncryptionService.disable();
		this.refresh();
	}
}

export const encryptionStore = new EncryptionStore();
