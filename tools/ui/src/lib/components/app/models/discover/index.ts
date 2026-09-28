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

/**
 * **ModelsDiscoverAvatar** - Org avatar for a model row
 *
 * Shows the org's avatar image, falling back to a monogram on a stable hue
 * derived from the org name when the image fails to load. Shared by the list,
 * the detail header and the model selector rows.
 */
export { default as ModelsDiscoverAvatar } from './ModelsDiscoverAvatar.svelte';

/**
 * **ModelsDiscoverDownloadProgressBar** - Thin download progress bar
 *
 * Normalizes bytes to a 0..100% bar; can pin to the bottom edge as an overlay.
 * Shared by the quant chips and the model selector's download rows.
 */
export { default as ModelsDiscoverDownloadProgressBar } from './ModelsDiscoverDownloadProgressBar.svelte';

export * from './ModelsDiscoverList';
export * from './ModelsDiscoverDetails';
