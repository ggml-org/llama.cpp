/** Presentation layer for Skills: labels, budget status, catalog filters, settings rows. */
import { BookOpen, List } from '@lucide/svelte';
import {
	buildSkillListToolDefinition,
	buildSkillReadToolDefinition,
	freezeSkillToolDefinition,
	SKILL_LIST_TOOL,
	SKILL_LIST_TOOL_DESCRIPTION,
	SKILL_READ_TOOL,
	SKILL_READ_TOOL_DESCRIPTION
} from '$lib/constants';
import type { SkillCatalogEntry, SkillPackedCatalog, SkillToolSetting } from '$lib/types';

// ---------------------------------------------------------------------------
// Provider labels
// ---------------------------------------------------------------------------

/** Server provider value for the provider-agnostic `.agents/` directory. */
const AGENTS_SKILL_PROVIDER = 'agents';

/** Tooltip for the provider-agnostic label. */
export const GENERIC_SKILL_PROVIDER_TOOLTIP =
	'This skill belongs to the .agents/ dir, which is provider agnostic';

/** True when the provider value is the provider-agnostic `.agents/` directory. */
export function isGenericSkillProvider(provider: string): boolean {
	return provider === AGENTS_SKILL_PROVIDER;
}

/** Map the server provider value `agents` to `generic`. */
export function skillProviderLabel(provider: string): string {
	return isGenericSkillProvider(provider) ? 'generic' : provider;
}

// ---------------------------------------------------------------------------
// Budget status chip
// ---------------------------------------------------------------------------

/**
 * Compact, diagnostic-list-shaped summary of the catalog budget status.
 * `null` means "the full catalog fits" or "nothing packed yet" - both
 * render nothing, matching the diagnostics block's minimal default.
 */
export interface SkillBudgetChip {
	label: 'Partial fit' | 'Tools disabled';
	detail: string;
}

export function deriveSkillBudgetChip(
	packed: SkillPackedCatalog | null,
	budget: number
): SkillBudgetChip | null {
	if (!packed) return null;

	const disabled = budget === 0 || packed.fullTokens === null;

	if (disabled) {
		return {
			detail:
				'Skills tools are disabled because the catalog budget is 0 tokens. The catalog stays available for browsing, but no Skills prompt envelope is packed into agentic runs.',
			label: 'Tools disabled'
		};
	}

	if (packed.included === packed.total) return null;

	const measureLabel = packed.estimated ? 'estimated' : 'exact';
	const fullTokensLabel = packed.fullTokens?.toLocaleString() ?? '';

	return {
		detail: `The full Skills catalog requires ${fullTokensLabel} tokens (${measureLabel}). ${packed.included} of ${packed.total} skills are included; list_skill() is available.`,
		label: 'Partial fit'
	};
}

// ---------------------------------------------------------------------------
// Catalog filters
// ---------------------------------------------------------------------------

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

/** Next excluded-provider set after toggling one provider's checkbox. */
export function toggleSkillProviderExclusion(
	excludedProviders: ReadonlySet<string>,
	provider: string,
	included: boolean
): Set<string> {
	const next = new Set(excludedProviders);

	if (included) next.delete(provider);
	else next.add(provider);

	return next;
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

// ---------------------------------------------------------------------------
// Settings rows
// ---------------------------------------------------------------------------

/** Settings-only rows for the Skills adapters. */
export const SKILL_TOOL_SETTINGS: readonly SkillToolSetting[] = Object.freeze([
	{
		definition: freezeSkillToolDefinition(buildSkillReadToolDefinition()),
		description: SKILL_READ_TOOL_DESCRIPTION,
		icon: BookOpen,
		key: `skill:${SKILL_READ_TOOL}`,
		label: 'Read skill',
		toolName: SKILL_READ_TOOL
	},
	{
		definition: freezeSkillToolDefinition(buildSkillListToolDefinition()),
		description: SKILL_LIST_TOOL_DESCRIPTION,
		icon: List,
		key: `skill:${SKILL_LIST_TOOL}`,
		label: 'List skills',
		toolName: SKILL_LIST_TOOL
	}
]);
