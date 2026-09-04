/**
 *
 * MODELS DISCOVER — DETAILS — DOWNLOAD OPTIONS
 *
 * The download area of the detail pane: GGUF files grouped by bit depth, each an
 * independent download action chip, plus the standalone terminal command preview.
 *
 */

/**
 * **ModelsDiscoverDetailsDownloadOptions** - GGUF download options
 *
 * Groups GGUF files by bit depth and renders one independent download
 * action chip per file, plus the standalone terminal command preview.
 */
export { default as ModelsDiscoverDetailsDownloadOptions } from './ModelsDiscoverDetailsDownloadOptions.svelte';

/**
 * **ModelsDiscoverDetailsDownloadOptionsRow** - One bit-depth group of quants
 *
 * A single bit-depth row inside ModelsDiscoverDetailsDownloadOptions: the
 * depth label with its memory hint and the quant chips of that depth.
 */
export { default as ModelsDiscoverDetailsDownloadOptionsRow } from './ModelsDiscoverDetailsDownloadOptionsRow.svelte';

/**
 * **ModelsDiscoverDetailsDownloadOptionsQuantDownloadButton** - One quant chip
 *
 * A single GGUF file as an independent action chip: download / retry when
 * idle, pause / resume / cancel while in flight, delete when downloaded.
 */
export { default as ModelsDiscoverDetailsDownloadOptionsQuantDownloadButton } from './ModelsDiscoverDetailsDownloadOptionsQuantDownloadButton.svelte';

/**
 * **ModelsDiscoverDetailsDownloadOptionsDownloadCommand** - Terminal command
 *
 * The `llama serve -hf ...` command box with inline quant selects and a copy
 * button; owns its picks, nothing two-way binds them to the quant chips.
 */
export { default as ModelsDiscoverDetailsDownloadOptionsDownloadCommand } from './ModelsDiscoverDetailsDownloadOptionsDownloadCommand.svelte';
