/**
 *
 * MODELS DISCOVER - DETAILS
 *
 * The right-hand detail pane of the discover view: header (avatar, name, stats, metadata
 * chips, capability badges), the download options area and the model-card README.
 *
 */

/** Detail pane of the selected model; fetches its own details and GGUF file list. */
export { default as ModelsDiscoverDetails } from './ModelsDiscoverDetails.svelte';

/** Fetches one repo's details, files and readme for hosts outside Discover. */
export { default as ModelsDiscoverDetailsContainer } from './ModelsDiscoverDetailsContainer.svelte';

/** Header of the detail pane. */
export { default as ModelsDiscoverDetailsHeader } from './ModelsDiscoverDetailsHeader.svelte';

/** Downloads, likes and last updated, formatted the way the Hub shows them. */
export { default as ModelsDiscoverDetailsHfHubStats } from './ModelsDiscoverDetailsHfHubStats.svelte';

/** Static placeholder matching the detail layout. */
export { default as ModelsDiscoverDetailsSkeleton } from './ModelsDiscoverDetailsSkeleton.svelte';

/** Model card README as markdown. */
export { default as ModelsDiscoverDetailsReadme } from './ModelsDiscoverDetailsReadme.svelte';

/** Metadata chips: size, context, architecture, license, chat template. */
export { default as ModelsDiscoverDetailsMetadata } from './ModelsDiscoverDetailsMetadata.svelte';

/** One label | value chip of the metadata row. */
export { default as ModelsDiscoverDetailsMetadataItem } from './ModelsDiscoverDetailsMetadataItem.svelte';

/** Chat template in a scrollable dialog with a copy button. */
export { default as ModelsDiscoverChatTemplateDialog } from './ModelsDiscoverChatTemplateDialog.svelte';

export * from './ModelsDiscoverDetailsDownloadOptions';
