/**
 * Read-only presentation of the server's GET /skills catalog.
 *
 * The route owns the CWD derivation and refresh lifecycle; this component
 * renders only contract-safe catalog fields with distinct loading, error,
 * unavailable, empty, and success states.
 */
export { default as SkillCatalog } from './SkillCatalog.svelte';
