/**
 *
 * MODELS DISCOVER - DETAILS - DOWNLOAD OPTIONS
 *
 * The download area of the detail pane: one chip per main quant, the drafts it rides with
 * badged on it, a draft checkbox and type pick at the bottom, and a terminal command
 * preview inside every row.
 *
 */

/** One download action per main quant, the sidecars it pulls, and the row's command. */
export { default as ModelsDiscoverDetailsDownloadOptions } from './ModelsDiscoverDetailsDownloadOptions.svelte';

/** One bit-depth row: the depth label opens its command, the quants of that depth download beside it. */
export { default as ModelsDiscoverDetailsDownloadOptionsRow } from './ModelsDiscoverDetailsDownloadOptionsRow.svelte';

/** A single GGUF file as an independent action chip. */
export { default as ModelsDiscoverDetailsDownloadOptionsQuantDownloadButton } from './ModelsDiscoverDetailsDownloadOptionsQuantDownloadButton.svelte';

/** Chip for a draft a downloaded model is missing; posts the model with that kind, so the server keeps what is on disk and pulls the draft. */
export { default as ModelsDiscoverDetailsDownloadOptionsDraftDownloadButton } from './ModelsDiscoverDetailsDownloadOptionsDraftDownloadButton.svelte';

/** The `llama serve -hf ...` command box of one row, with a draft type pick and a copy button. */
export { default as ModelsDiscoverDetailsDownloadOptionsDownloadCommand } from './ModelsDiscoverDetailsDownloadOptionsDownloadCommand.svelte';
