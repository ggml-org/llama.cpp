/**
 *
 * MODELS HUB
 *
 * Components for the Models Hub route (`/models-hub`): a sidebar list of
 * HuggingFace GGUF models and a detail view for the selected model.
 *
 */

/**
 * **ModelsDiscover** - Models hub explorer
 *
 * The complete discovery layout: a sidebar search + model list on the left and a
 * detail view for the selected model on the right. Used as the body of the
 * discovery dialog.
 */
export { default as ModelsDiscover } from './ModelsDiscover.svelte';

/**
 * **ModelsDiscoverList** - Sidebar model list
 *
 * Renders the hub's model list as a navigable column. Each row links to the
 * model's detail route and highlights the active one.
 */
export { default as ModelsDiscoverList } from './ModelsDiscoverList.svelte';

/**
 * **ModelsDiscoverListSearch** - Sidebar search input
 *
 * Debounced search field for the model list.
 */
export { default as ModelsDiscoverListSearch } from './ModelsDiscoverListSearch.svelte';

/**
 * **ModelsDiscoverItem** - Single sidebar row
 *
 * One model entry in the sidebar list. Links to `/models-hub/[org]/[model]`.
 */
export { default as ModelsDiscoverListItem } from './ModelsDiscoverListItem.svelte';

/**
 * **ModelsDiscoverListItemSkeleton** - Skeleton sidebar row
 *
 * Pulsing placeholder matching a ModelsDiscoverListItem row, shown while the
 * list is loading.
 */
export { default as ModelsDiscoverListItemSkeleton } from './ModelsDiscoverListItemSkeleton.svelte';

/**
 * **ModelsDiscoverAvatar** - Org avatar for a model row
 *
 * Shows the org's avatar image, falling back to a monogram on a stable hue
 * derived from the org name when the image fails to load.
 */
export { default as ModelsDiscoverAvatar } from './ModelsDiscoverAvatar.svelte';

/**
 * **ModelsDiscoverDetails** - Model detail view
 *
 * Detail pane for the selected model. Loads its own data (details + GGUF file
 * list) from HuggingFaceService based on the `modelId` route param.
 */
export { default as ModelsDiscoverModelDetails } from './ModelsDiscoverModelDetails.svelte';

/**
 * **ModelsDiscoverDetailsHeader** - Detail view header
 *
 * Shows the model avatar (base org + quant org corner badge), name, base model
 * info, stats, metadata chips and capability badges.
 */
export { default as ModelsDiscoverModelDetailsHeader } from './ModelsDiscoverModelDetailsHeader.svelte';

/**
 * **ModelsDiscoverDetailsDownloadOptions** - GGUF download options
 *
 * Groups GGUF files by bit depth and renders per-file download buttons with
 * progress, owned by the download confirmation dialog.
 */
export { default as ModelsDiscoverModelDetailsDownloadOptions } from './ModelsDiscoverModelDetailsDownloadOptions.svelte';

/**
 * **ModelsDiscoverDetailsDownloadOptionsRow** - One bit-depth group of quants
 *
 * A single bit-depth row inside ModelsDiscoverModelDetailsDownloadOptions: the
 * depth label with its memory hint and the quant chips of that depth.
 */
export { default as ModelsDiscoverModelDetailsDownloadOptionsRow } from './ModelsDiscoverModelDetailsDownloadOptionsRow.svelte';

/**
 * **ModelsDiscoverDetailsDownloadOptionsQuantToggle** - One quant chip
 *
 * A single GGUF file as a toggle chip inside the download options toggle
 * group, or as a static done chip when the file is already downloaded.
 */
export { default as ModelsDiscoverModelDetailsDownloadOptionsQuantToggle } from './ModelsDiscoverModelDetailsDownloadOptionsQuantToggle.svelte';

/**
 * **ModelsDiscoverDetailsDownloadOptionsDownloadButton** - Download CTA
 *
 * Full-width primary button that queues the current selection for download.
 */
export { default as ModelsDiscoverModelDetailsDownloadOptionsDownloadButton } from './ModelsDiscoverModelDetailsDownloadOptionsDownloadButton.svelte';

/**
 * **ModelsDiscoverDetailsDownloadOptionsDownloadCommand** - Terminal command
 *
 * The `llama serve -hf ...` command box with inline quant selects and a copy
 * button; the quant picks are delegated back to the parent via callbacks.
 */
export { default as ModelsDiscoverModelDetailsDownloadOptionsDownloadCommand } from './ModelsDiscoverModelDetailsDownloadOptionsDownloadCommand.svelte';

/**
 * **ModelsDiscoverDetailsSkeleton** - Detail view loading skeleton
 *
 * Static placeholder matching the detail layout: header with avatar and name,
 * metadata chips, the download options box and readme text lines.
 */
export { default as ModelsDiscoverModelDetailsSkeleton } from './ModelsDiscoverModelDetailsSkeleton.svelte';

/**
 * **ModelsDiscoverDetailsMetadataItem** - Single metadata chip
 *
 * One label | value chip of the detail view's metadata row (model size,
 * context, architecture, license).
 */
export { default as ModelsDiscoverModelDetailsMetadataItem } from './ModelsDiscoverModelDetailsMetadataItem.svelte';

/**
 * **ModelsDiscoverChatTemplateDialog** - Chat template viewer
 *
 * Shows the model's chat template in a scrollable dialog with a copy button.
 */
export { default as ModelsDiscoverChatTemplateDialog } from './ModelsDiscoverChatTemplateDialog.svelte';

/**
 * **ModelsDiscoverDetailsReadme** - Detail view README
 *
 * Renders the model card README as markdown.
 */
export { default as ModelsDiscoverModelDetailsReadme } from './ModelsDiscoverModelDetailsReadme.svelte';

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
