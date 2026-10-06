/** One family of a model list, and the entries it covers. */
export interface ModelFamilyGroup<T> {
	entries: T[];
	key: string;
	label: string;
}

/** Leading letters of a repo name, e.g. the `Qwen` of `Qwen3.8-27B`. */
const FAMILY_LETTERS = /^[A-Za-z]+/;
/** Separators between the segments of a repo name. */
const FAMILY_SEPARATORS = /[-_.]/;

/**
 * Family a repo belongs to: the leading letters of its name, so `Qwen3.5` and
 * `Qwen3.8-27B` both read as `Qwen`.
 */
export function modelFamilyKey(repo: string): string {
	const name = repo.split('/').pop() ?? repo;
	const letters = name.match(FAMILY_LETTERS);

	return letters ? letters[0] : name.split(FAMILY_SEPARATORS)[0] || name;
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
