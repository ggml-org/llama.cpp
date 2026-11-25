/**
 * Memoizes `fn` by a string key, holding at most `maxSize` results: once full
 * the cache is emptied, which is enough for values that are cheap to rebuild.
 * `keyOf` returns null for a call that should not be cached.
 */
export function memoizeBounded<Args extends unknown[], Result>(
	fn: (...args: Args) => Result,
	keyOf: (...args: Args) => string | null,
	maxSize: number
): (...args: Args) => Result {
	const cache = new Map<string, Result>();

	return (...args) => {
		const key = keyOf(...args);

		if (key === null) return fn(...args);

		if (cache.has(key)) return cache.get(key) as Result;

		const result = fn(...args);

		if (cache.size >= maxSize) cache.clear();

		cache.set(key, result);

		return result;
	};
}
