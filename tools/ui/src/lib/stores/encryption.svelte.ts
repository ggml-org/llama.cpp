/**
 * Encryption Store - reactive facade over EncryptionService
 *
 * Exposes the encryption session state (enabled/unlocked) as reactive state
 * and serializes the app bootstrap behind the unlock gate: `init()` awaits
 * `ensureUnlocked()`, which only resolves once the user has unlocked (or when
 * no unlock is needed), so no conversation data is ever loaded while locked.
 *
 * Also manages an auto-lock idle timer: after a configurable period of inactivity
 * the session key is dropped, requiring the user to re-enter their passphrase.
 */

import { EncryptionService } from '$lib/services/encryption.service';
import {
	ENCRYPTION_IDLE_TIMEOUT_LOCALSTORAGE_KEY,
	ENCRYPTION_SESSION_LOCALSTORAGE_KEY
} from '$lib/constants';
import { mcpStore } from '$lib/stores/mcp.svelte';

/** Default idle timeout in minutes */
const DEFAULT_IDLE_TIMEOUT_MINUTES = 5;

const IDLE_TIMEOUT_PRESETS = [
	{ value: 0, label: 'Never' },
	{ value: 1, label: '1 minute' },
	{ value: 5, label: '5 minutes' },
	{ value: 15, label: '15 minutes' },
	{ value: 30, label: '30 minutes' },
	{ value: 60, label: '1 hour' }
] as const;

export type IdleTimeoutMinutes = (typeof IDLE_TIMEOUT_PRESETS)[number]['value'];

/** Allowed timeout presets in minutes; the 1-minute preset is dev-only, for testing auto-lock */
export const IDLE_TIMEOUT_OPTIONS = IDLE_TIMEOUT_PRESETS.filter(
	(option) => import.meta.env.DEV || option.value !== 1
);

/** Persisted resumable session; the raw DEK trades at-rest protection for convenience */
interface SessionRecord {
	dek: string;
	lastActivity: number;
}

/** Activity bumps rewrite the session record at most this often */
const SESSION_WRITE_THROTTLE_MS = 30_000;

class EncryptionStore {
	isSupported = $state(false);
	isEnabled = $state(false);
	isUnlocked = $state(false);

	needsUnlock = $derived(this.isEnabled && !this.isUnlocked);

	/** Idle timeout in minutes (0 = never auto-lock) */
	idleTimeoutMinutes = $state<IdleTimeoutMinutes>(this.loadIdleTimeout());

	/** Timestamp when the idle timer will lock the session; null when no lock is scheduled */
	idleDeadlineAt = $state<number | null>(null);

	private unlockPromise: Promise<void> | null = null;
	private unlockResolver: (() => void) | null = null;

	private idleTimer: ReturnType<typeof setTimeout> | null = null;
	private lastSessionWriteAt = 0;
	private lockListeners: (() => void)[] = [];
	private unlockListeners: (() => void)[] = [];

	refresh(): void {
		this.isSupported = EncryptionService.isSupported();
		this.isEnabled = EncryptionService.isEnabled();
		this.isUnlocked = EncryptionService.isUnlocked();

		// Start / stop idle timer based on lock state
		if (this.isUnlocked && this.isEnabled) {
			this.resetIdleTimer();
		} else {
			this.clearIdleTimer();
		}
	}

	/**
	 * Resolves immediately unless encryption is enabled and locked; otherwise
	 * blocks until {@link unlockWithPassphrase} succeeds.
	 */
	ensureUnlocked(): Promise<void> {
		this.refresh();
		if (!this.isEnabled || this.isUnlocked) return Promise.resolve();

		return this.resumeThenGate();
	}

	// A persisted session within the idle window unlocks without a passphrase;
	// otherwise fall through to the shared gate
	private async resumeThenGate(): Promise<void> {
		await this.tryResumeSession();
		if (!this.isEnabled || this.isUnlocked) return;

		// Concurrent callers share a single gate
		if (!this.unlockPromise) {
			this.unlockPromise = new Promise((resolve) => {
				this.unlockResolver = resolve;
			});
		}

		return this.unlockPromise;
	}

	private async tryResumeSession(): Promise<void> {
		if (typeof localStorage === 'undefined' || !this.isEnabled || this.isUnlocked) return;

		try {
			const raw = localStorage.getItem(ENCRYPTION_SESSION_LOCALSTORAGE_KEY);
			if (!raw) return;

			const record = JSON.parse(raw) as Partial<SessionRecord>;
			const idleMs = this.idleTimeoutMinutes * 60 * 1000;
			const withinWindow =
				typeof record.lastActivity === 'number' &&
				(idleMs === 0 || Date.now() - record.lastActivity <= idleMs);

			if (typeof record.dek !== 'string' || !withinWindow) {
				this.clearSession();
				return;
			}

			await EncryptionService.importRawDek(record.dek);
			this.refresh();
		} catch {
			this.clearSession();
		}
	}

	private async persistSession(): Promise<void> {
		if (typeof localStorage === 'undefined') return;

		const dek = await EncryptionService.exportRawDek();
		if (!dek) return;

		this.lastSessionWriteAt = Date.now();
		const record: SessionRecord = { dek, lastActivity: this.lastSessionWriteAt };
		localStorage.setItem(ENCRYPTION_SESSION_LOCALSTORAGE_KEY, JSON.stringify(record));
	}

	private clearSession(): void {
		if (typeof localStorage === 'undefined') return;
		localStorage.removeItem(ENCRYPTION_SESSION_LOCALSTORAGE_KEY);
	}

	async unlockWithPassphrase(passphrase: string): Promise<boolean> {
		const unlocked = await EncryptionService.unlockWithPassphrase(passphrase);
		if (unlocked) {
			this.refresh();
			await mcpStore.loadSecrets();
			this.unlockResolver?.();
			this.unlockResolver = null;
			this.unlockPromise = null;
			await this.persistSession();
			for (const listener of this.unlockListeners) listener();
		}
		return unlocked;
	}

	async setupWithPassphrase(passphrase: string): Promise<void> {
		await EncryptionService.setupWithPassphrase(passphrase);
		this.refresh();
		await this.persistSession();
	}

	async changePassphrase(current: string, next: string): Promise<boolean> {
		return await EncryptionService.changePassphrase(current, next);
	}

	/**
	 * Signal that the user is still active. Resets the idle timer.
	 * Called by the layout on mouse / keyboard / scroll events.
	 */
	onUserActivity(): void {
		if (!this.isUnlocked) return;

		this.resetIdleTimer();

		if (Date.now() - this.lastSessionWriteAt > SESSION_WRITE_THROTTLE_MS) {
			void this.persistSession();
		}
	}

	/**
	 * Update the idle timeout preference and persist it.
	 */
	setIdleTimeout(minutes: IdleTimeoutMinutes): void {
		this.idleTimeoutMinutes = minutes;
		this.saveIdleTimeout(minutes);

		// If unlocked, restart the timer with the new value
		if (this.isUnlocked) {
			this.resetIdleTimer();
		}
	}

	/** Register a callback fired after the session locks (idle timeout or manual) */
	onLock(listener: () => void): () => void {
		this.lockListeners.push(listener);
		return () => {
			this.lockListeners = this.lockListeners.filter((l) => l !== listener);
		};
	}

	/** Register a callback fired after a successful passphrase unlock */
	onUnlock(listener: () => void): () => void {
		this.unlockListeners.push(listener);
		return () => {
			this.unlockListeners = this.unlockListeners.filter((l) => l !== listener);
		};
	}

	/** Drops the session DEK; persisted data stays encrypted */
	lock(): void {
		this.clearIdleTimer();
		this.clearSession();
		EncryptionService.lock();
		this.refresh();
		for (const listener of this.lockListeners) listener();
	}

	disable(): void {
		this.clearIdleTimer();
		this.clearSession();
		EncryptionService.disable();
		this.refresh();
		this.unlockResolver?.();
		this.unlockResolver = null;
		this.unlockPromise = null;
	}

	/* ── Idle timer internals ─────────────────────────────────── */

	private resetIdleTimer(): void {
		this.clearIdleTimer();

		const timeoutMs = this.idleTimeoutMinutes * 60 * 1000;
		if (timeoutMs <= 0) return; // 0 = never auto-lock

		this.idleDeadlineAt = Date.now() + timeoutMs;
		this.idleTimer = setTimeout(() => {
			this.lock();
		}, timeoutMs);
	}

	private clearIdleTimer(): void {
		if (this.idleTimer !== null) {
			clearTimeout(this.idleTimer);
			this.idleTimer = null;
		}
		this.idleDeadlineAt = null;
	}

	private loadIdleTimeout(): IdleTimeoutMinutes {
		if (typeof localStorage === 'undefined') return DEFAULT_IDLE_TIMEOUT_MINUTES;
		try {
			const raw = localStorage.getItem(ENCRYPTION_IDLE_TIMEOUT_LOCALSTORAGE_KEY);
			if (raw === null) return DEFAULT_IDLE_TIMEOUT_MINUTES;
			const parsed = Number(raw);
			const validValues = new Set<number>(IDLE_TIMEOUT_OPTIONS.map((o) => o.value));
			return validValues.has(parsed)
				? (parsed as IdleTimeoutMinutes)
				: DEFAULT_IDLE_TIMEOUT_MINUTES;
		} catch {
			return DEFAULT_IDLE_TIMEOUT_MINUTES;
		}
	}

	private saveIdleTimeout(minutes: IdleTimeoutMinutes): void {
		if (typeof localStorage === 'undefined') return;
		localStorage.setItem(ENCRYPTION_IDLE_TIMEOUT_LOCALSTORAGE_KEY, String(minutes));
	}
}

export const encryptionStore = new EncryptionStore();
