import { USER_PREFERENCE_KEYS } from '$lib/constants';
import { PreferenceScope } from '$lib/enums';

/**
 * PreferencesService - single access point for persisted key/value preferences
 *
 * Mirrors the localStorage API, errors included, so callers keep their own
 * handling. Every key is either the user's own data or device-local UI state
 * (see {@link PreferencesService.scopeOf}); both live in localStorage for now.
 *
 * Legacy key migrations in MigrationService read localStorage directly, since
 * they are about this browser's past data.
 */
export class PreferencesService {
	static getItem(key: string): string | null {
		return localStorage.getItem(key);
	}

	static removeItem(key: string): void {
		localStorage.removeItem(key);
	}

	/** Whether a key holds the user's own data or this browser's UI state. */
	static scopeOf(key: string): PreferenceScope {
		return USER_PREFERENCE_KEYS.has(key) ? PreferenceScope.USER : PreferenceScope.DEVICE;
	}

	static setItem(key: string, value: string): void {
		localStorage.setItem(key, value);
	}
}
