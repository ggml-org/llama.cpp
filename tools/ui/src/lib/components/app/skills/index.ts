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
 * Route-local read preview of one selected skill: rendered body vs raw source
 * with Markdown/Raw modes and stale-read suppression. Never creates messages
 * or activation records.
 */
export { default as SkillDetail } from './SkillDetail.svelte';
