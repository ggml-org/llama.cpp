/** One family of a model list, and the entries it covers. */
export interface ModelFamilyGroup<T> {
	entries: T[];
	key: string;
	label: string;
}

/**
 * Family a repo belongs to, from its name. The version is dropped, so `Qwen3.5`,
 * `Qwen3.8-27B` and `Qwen3.8-Flash-Next` all read as `Qwen`. A name that does not
 * start with letters keeps its first segment.
 */
export function modelFamilyKey(repo: string): string {
	const name = repo.split('/').pop() ?? repo;
	const letters = name.match(/^[A-Za-z]+/);

	return letters ? letters[0] : (name.split(/[-_.]/)[0] ?? name);
}

/** Fold entries into families, so `Qwen` collects its sizes and variants. */
export function groupModelFamilies<T>(
	entries: T[],
	modelOf: (entry: T) => string
): ModelFamilyGroup<T>[] {
	const families = new Map<string, ModelFamilyGroup<T>>();

	for (const entry of entries) {
		const key = modelFamilyKey(modelOf(entry));
		const family = families.get(key);

		if (family) {
			family.entries.push(entry);

			continue;
		}

		families.set(key, { entries: [entry], key, label: key });
	}

	return Array.from(families.values());
}
