/**
 * Read-only presentation of the server's GET /skills catalog.
 *
 * The route owns the CWD derivation and refresh lifecycle; this component
 * renders only contract-safe catalog fields with distinct loading, error,
 * unavailable, empty, and success states.
 */
export { default as SkillCatalog } from './SkillCatalog.svelte';

/**
 * Budget status of the ready catalog, packed with the same policy as agentic
 * runs. Copy derives only from returned SkillPackedCatalog values and the
 * configured budget, with distinct disabled / complete / partial semantics.
 */
export { default as SkillBudgetStatus } from './SkillBudgetStatus.svelte';

/**
 * Keyboard-operable catalog cards. Renders only contract-safe fields; the
 * opaque catalog XML is never rendered here.
 */
export { default as SkillCatalogList } from './SkillCatalogList.svelte';

/**
 * Shared visible provider label: maps the API value `agents` to `generic`
 * with the exact provider-agnostic tooltip on hover and keyboard focus;
 * every other provider renders unchanged without a tooltip. Presentation
 * only - the server value and identity stay unchanged.
 */
export { default as SkillProviderLabel } from './SkillProviderLabel.svelte';

/**
 * Route-local read preview of one selected skill: rendered body vs raw source
 * with Markdown/Raw modes and stale-read suppression. Never creates messages
 * or activation records.
 */
export { default as SkillDetail } from './SkillDetail.svelte';

export { default as SkillResourcePicker } from './SkillResourcePicker.svelte';
export { default as SkillResourcePreview } from './SkillResourcePreview.svelte';
