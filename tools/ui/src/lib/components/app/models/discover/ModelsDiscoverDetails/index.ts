/**
 *
 * MODELS DISCOVER — DETAILS
 *
 * The right-hand detail pane of the discover view: header (avatar, name, stats, metadata
 * chips, capability badges), the download options area and the model-card README.
 *
 */

/**
 * **ModelsDiscoverDetails** - Model detail view
 *
 * Detail pane for the selected model. Loads its own data (details + GGUF file
 * list) from HuggingFaceService based on the `modelId` route param.
 */
export { default as ModelsDiscoverDetails } from './ModelsDiscoverDetails.svelte';

/**
 * **ModelsDiscoverDetailsHeader** - Detail view header
 *
 * Shows the model avatar (base org + quant org corner badge), name, base model
 * info, stats, metadata chips and capability badges.
 */
export { default as ModelsDiscoverDetailsHeader } from './ModelsDiscoverDetailsHeader.svelte';

/**
 * **ModelsDiscoverDetailsSkeleton** - Detail view loading skeleton
 *
 * Static placeholder matching the detail layout: header with avatar and name,
 * metadata chips, the download options box and readme text lines.
 */
export { default as ModelsDiscoverDetailsSkeleton } from './ModelsDiscoverDetailsSkeleton.svelte';

/**
 * **ModelsDiscoverDetailsReadme** - Detail view README
 *
 * Renders the model card README as markdown.
 */
export { default as ModelsDiscoverDetailsReadme } from './ModelsDiscoverDetailsReadme.svelte';

/**
 * **ModelsDiscoverDetailsMetadata** - Detail view metadata row
 *
 * The detail view's metadata chips (model size, context, architecture, license,
 * chat template).
 */
export { default as ModelsDiscoverDetailsMetadata } from './ModelsDiscoverDetailsMetadata.svelte';

/**
 * **ModelsDiscoverDetailsMetadataItem** - Single metadata chip
 *
 * One label | value chip of the detail view's metadata row (model size,
 * context, architecture, license).
 */
export { default as ModelsDiscoverDetailsMetadataItem } from './ModelsDiscoverDetailsMetadataItem.svelte';

/**
 * **ModelsDiscoverChatTemplateDialog** - Chat template viewer
 *
 * Shows the model's chat template in a scrollable dialog with a copy button.
 */
export { default as ModelsDiscoverChatTemplateDialog } from './ModelsDiscoverChatTemplateDialog.svelte';
