/**
 *
 * MODELS HUB
 *
 * Components for the Models Hub route (`/models-hub`): a sidebar list of
 * HuggingFace GGUF models and a detail view for the selected model.
 *
 */

/**
 * **ModelsHubList** - Sidebar model list
 *
 * Renders the hub's model list as a navigable column. Each row links to the
 * model's detail route and highlights the active one.
 */
export { default as ModelsHubList } from './ModelsHubList.svelte';

/**
 * **ModelsHubListItem** - Single sidebar row
 *
 * One model entry in the sidebar list. Links to `/models-hub/[org]/[model]`.
 */
export { default as ModelsHubListItem } from './ModelsHubListItem.svelte';

/**
 * **ModelsHubModelDetails** - Model detail view
 *
 * Detail pane for the selected model. Loads its own data (details + GGUF file
 * list) from HuggingFaceService based on the `modelId` route param.
 */
export { default as ModelsHubModelDetails } from './ModelsHubModelDetails.svelte';

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
