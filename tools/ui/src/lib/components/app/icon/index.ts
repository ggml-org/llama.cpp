/**
 *
 * ICON
 *
 * Generic icon helpers keyed by string name (e.g. lucide icon names returned
 * programmatically from services). Currently hosts a single component that
 * resolves a name to a lucide icon.
 *
 */

/**
 * **IconFromName** - Resolve a string icon name to a rendered lucide icon.
 *
 * Use this when an upstream API gives you an icon identifier and you don't
 * want to write a per-case `{#if name === 'x'}<X />{/if}` ladder. Falls back
 * to a generic circle when the name isn't recognised.
 */
export { default as IconFromName } from './IconFromName.svelte';
