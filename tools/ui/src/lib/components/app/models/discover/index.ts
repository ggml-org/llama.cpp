/**
 *
 * MODELS DISCOVER
 *
 * Components for the Models Discover view: a sidebar search + list of
 * HuggingFace GGUF models and a detail view for the selected model, used as the
 * body of the discovery dialog. The list and detail trees live in their own
 * subfolders; this barrel re-exports them alongside the shared leaves.
 *
 */

/**
 * **ModelsDiscover** - Models discover explorer
 *
 * The complete discovery layout: a sidebar search + model list on the left and a
 * detail view for the selected model on the right. Used as the body of the
 * discovery dialog.
 */
export { default as ModelsDiscover } from './ModelsDiscover.svelte';

export * from './ModelsDiscoverList';
export * from './ModelsDiscoverDetails';
