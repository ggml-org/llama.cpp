import type { SkillPackedCatalog } from '$lib/types';

/**
 * Compact, diagnostic-list-shaped summary of the catalog budget status.
 * `null` means "the full catalog fits" or "nothing packed yet" — both
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
