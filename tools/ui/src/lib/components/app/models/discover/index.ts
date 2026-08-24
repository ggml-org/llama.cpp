/**
 *
 * MODELS HUB
 *
 * Components for the Models Hub route (`/models-hub`): a sidebar list of
 * HuggingFace GGUF models and a detail view for the selected model.
 *
 */

/**
 * **ModelsDiscoverList** - Sidebar model list
 *
 * Renders the hub's model list as a navigable column. Each row links to the
 * model's detail route and highlights the active one.
 */
export { default as ModelsDiscoverList } from './ModelsDiscoverList.svelte';

/**
 * **ModelsDiscoverItem** - Single sidebar row
 *
 * One model entry in the sidebar list. Links to `/models-hub/[org]/[model]`.
 */
export { default as ModelsDiscoverItem } from './ModelsDiscoverItem.svelte';

/**
 * **ModelsDiscoverAvatar** - Org avatar for a model row
 *
 * Shows the org's avatar image, falling back to a monogram on a stable hue
 * derived from the org name when the image fails to load.
 */
export { default as ModelsDiscoverAvatar } from './ModelsDiscoverAvatar.svelte';

/**
 * **ModelsDiscoverInfo** - Model name + metadata for a row
 *
 * Renders the model name via ModelId and the remaining metadata (org, last
 * modified, params, downloads, likes, vision) below it.
 */
export { default as ModelsDiscoverInfo } from './ModelsDiscoverInfo.svelte';

/**
 * **ModelsDiscoverDetails** - Model detail view
 *
 * Detail pane for the selected model. Loads its own data (details + GGUF file
 * list) from HuggingFaceService based on the `modelId` route param.
 */
export { default as ModelsDiscoverDetails } from './ModelsDiscoverDetails.svelte';

/**
 * **DialogModelDownload** - Download confirmation / progress dialog
 *
 * Confirms a single GGUF download, tracks live progress over the SSE feed and
 * offers cancel / delete-&-retry flows.
 */
export { default as DialogModelDownload } from './DialogModelDownload.svelte';

/**
 * **DownloadProgressBar** - Thin download progress bar
 *
 * Normalizes bytes to a 0..100% bar; can pin to the bottom edge as an overlay.
 */
export { default as DownloadProgressBar } from './DownloadProgressBar.svelte';
