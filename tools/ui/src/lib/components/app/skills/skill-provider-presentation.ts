/**
 * Presentation mapping for the server-authoritative skill provider value.
 *
 * The API and all underlying data continue to use `provider: "agents"` for
 * skills discovered from `.agents/skills`; only user-visible surfaces map it
 * to the display label `generic`. Resolution precedence, diagnostics, opaque
 * identity construction, stored activation metadata, and compatibility
 * behavior stay unchanged.
 */

/** Exact hover / keyboard-focus explanation for the provider-agnostic label. */
export const GENERIC_SKILL_PROVIDER_TOOLTIP =
	'This skill belongs to the .agents/ dir, which is provider agnostic';

/** Map the API provider value `agents` to the display label `generic`. */
export function skillProviderLabel(provider: string): string {
	return provider === 'agents' ? 'generic' : provider;
}