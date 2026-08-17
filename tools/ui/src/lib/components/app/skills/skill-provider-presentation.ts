/** Maps server provider values to user-facing labels. */

/** Tooltip for the provider-agnostic label. */
export const GENERIC_SKILL_PROVIDER_TOOLTIP =
	'This skill belongs to the .agents/ dir, which is provider agnostic';

/** Map the server provider value `agents` to `generic`. */
export function skillProviderLabel(provider: string): string {
	return provider === 'agents' ? 'generic' : provider;
}