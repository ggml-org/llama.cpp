import { browser } from '$app/environment';

type PersistedValue<T> = {
	get value(): T;
	set value(newValue: T);
};

export function persisted<T>(key: string, initialValue: T): PersistedValue<T> {
	let value = initialValue;

	if (browser) {
		try {
			const stored = localStorage.getItem(key);

			if (stored !== null) {
				value = JSON.parse(stored) as T;
			}
		} catch (error) {
			console.warn(`Failed to load ${key}:`, error);
		}
	}

	const persist = (next: T) => {
		if (!browser) {
			return;
		}

		try {
			if (next === null || next === undefined) {
				localStorage.removeItem(key);
				return;
			}

			localStorage.setItem(key, JSON.stringify(next));
		} catch (error) {
			console.warn(`Failed to persist ${key}:`, error);
		}
	};

	return {
		get value() {
			return value;
		},

		set value(newValue: T) {
			value = newValue;
			persist(newValue);
		}
	};
}
