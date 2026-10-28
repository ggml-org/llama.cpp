import { USER_PREFERENCE_KEYS } from '$lib/constants';
import { PreferenceScope } from '$lib/enums';

/** Sends a batch of changed user preferences to where they are stored remotely */
export type PreferencesPush = (set: Record<string, string>, remove: string[]) => Promise<void>;

/**
 * PreferencesService - single access point for persisted key/value preferences
 *
 * Mirrors the localStorage API, errors included, so callers keep their own
 * handling. Every key is either the user's own data or device-local UI state
 * (see {@link PreferencesService.scopeOf}). Device-local keys always live in
 * localStorage; user data does too unless {@link PreferencesService.useRemote}
 * moved it to the server, in which case it is read from memory and written
 * back in batches.
 *
 * Legacy key migrations in MigrationService read localStorage directly, since
 * they are about this browser's past data.
 */
export class PreferencesService {
	private static outbox = { remove: new Set<string>(), set: new Map<string, string>() };
	private static push: PreferencesPush | null = null;
	private static pushing: Promise<void> = Promise.resolve();
	private static pushScheduled = false;
	private static remote: Map<string, string> | null = null;

	/** Whether user data is currently read from and written to the server. */
	static get isRemote(): boolean {
		return this.remote !== null;
	}

	static getItem(key: string): string | null {
		if (this.isRemoteKey(key)) return this.remote!.get(key) ?? null;

		return localStorage.getItem(key);
	}

	static removeItem(key: string): void {
		if (this.isRemoteKey(key)) {
			this.remote!.delete(key);
			this.outbox.set.delete(key);
			this.outbox.remove.add(key);
			this.schedulePush();

			return;
		}

		localStorage.removeItem(key);
	}

	/** Whether a key holds the user's own data or this browser's UI state. */
	static scopeOf(key: string): PreferenceScope {
		return USER_PREFERENCE_KEYS.has(key) ? PreferenceScope.USER : PreferenceScope.DEVICE;
	}

	static setItem(key: string, value: string): void {
		if (this.isRemoteKey(key)) {
			this.remote!.set(key, value);
			this.outbox.remove.delete(key);
			this.outbox.set.set(key, value);
			this.schedulePush();

			return;
		}

		localStorage.setItem(key, value);
	}

	/** Serve user data from `values` (loaded from the server) and send changes through `push`. */
	static useRemote(values: Record<string, string>, push: PreferencesPush): void {
		this.remote = new Map(Object.entries(values));
		this.push = push;
	}

	private static isRemoteKey(key: string): boolean {
		return this.remote !== null && this.scopeOf(key) === PreferenceScope.USER;
	}

	/** Coalesce writes made in the same task into one request, sent in order. */
	private static schedulePush(): void {
		if (this.pushScheduled) return;

		this.pushScheduled = true;
		queueMicrotask(() => {
			this.pushScheduled = false;

			const set = Object.fromEntries(this.outbox.set);
			const remove = [...this.outbox.remove];
			const push = this.push;

			this.outbox = { remove: new Set(), set: new Map() };

			if (!push) return;

			this.pushing = this.pushing.then(() => push(set, remove)).catch(() => {});
		});
	}
}
