/**
 * What the package keeps in localStorage. Every access is allowed to fail, for there may be no
 * storage (outside a browser, or blocked), and the value then lives for this session only.
 */

export function readStorage(key: string): string | null {
	try {
		return localStorage.getItem(key);
	} catch {
		// no storage, e.g. outside a browser
		return null;
	}
}

export function writeStorage(key: string, value: string | null): void {
	try {
		if (value === null) localStorage.removeItem(key);
		else localStorage.setItem(key, value);
	} catch {
		// storage full or blocked
	}
}

/** The JSON a key holds, or `null` when there is none or it is not JSON. */
export function readJsonStorage(key: string): unknown {
	try {
		return JSON.parse(readStorage(key) ?? 'null');
	} catch {
		// not what the package wrote
		return null;
	}
}

export function writeJsonStorage(key: string, value: unknown): void {
	writeStorage(key, JSON.stringify(value));
}
