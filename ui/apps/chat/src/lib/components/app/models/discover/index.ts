/**
 *
 * MODELS DISCOVER
 *
 * Components for the Models Discover view: a sidebar search + list of HuggingFace GGUF
 * models and a detail view for the selected model, used as the body of the discovery
 * dialog. The list and detail trees live in their own subfolders; this barrel re-exports
 * them alongside the shared leaves.
 *
 */

/** The whole view: sidebar list on the left, detail pane on the right. */
export { default as ModelsDiscover } from './ModelsDiscover.svelte';

export * from './ModelsDiscoverList';
export * from './ModelsDiscoverDetails';
