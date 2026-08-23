import type { SkillCatalogEntry } from '$lib/types';

/** Search text + facet state for the Skills catalog toolbar. */
export interface SkillCatalogFilters {
	query: string;
	excludedProviders: ReadonlySet<string>;
	includeProject: boolean;
}

type QueryRank = 'name' | 'description' | null;

function rankByQuery(entry: SkillCatalogEntry, query: string): QueryRank {
	if (entry.name.toLowerCase().includes(query)) return 'name';
	if (entry.description.toLowerCase().includes(query)) return 'description';

	return null;
}

/**
 * Filters entries by provider/scope facets, then by search text. Name
 * matches rank above description matches; an empty query returns the
 * facet-filtered entries in their original order with no re-sort.
 */
export function applySkillCatalogFilters(
	entries: readonly SkillCatalogEntry[],
	filters: SkillCatalogFilters
): SkillCatalogEntry[] {
	const facetFiltered = entries.filter((entry) => {
		if (!filters.includeProject && entry.scope === 'project') return false;
		if (filters.excludedProviders.has(entry.provider)) return false;

		return true;
	});

	const query = filters.query.trim().toLowerCase();
	if (!query) return facetFiltered;

	const nameMatches: SkillCatalogEntry[] = [];
	const descriptionMatches: SkillCatalogEntry[] = [];

	for (const entry of facetFiltered) {
		const rank = rankByQuery(entry, query);

		if (rank === 'name') nameMatches.push(entry);
		else if (rank === 'description') descriptionMatches.push(entry);
	}

	return [...nameMatches, ...descriptionMatches];
}

/** Sorted, de-duplicated provider list present in the given entries. */
export function distinctSkillProviders(entries: readonly SkillCatalogEntry[]): string[] {
	return Array.from(new Set(entries.map((entry) => entry.provider))).sort();
}
