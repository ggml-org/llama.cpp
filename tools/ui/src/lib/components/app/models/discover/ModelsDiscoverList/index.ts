/**
 *
 * MODELS DISCOVER - LIST
 *
 * The discover sidebar column: a debounced search field above a navigable list of
 * HuggingFace GGUF models, with skeleton rows while loading.
 *
 */

/**
 * **ModelsDiscoverList** - Sidebar model list
 *
 * Renders the discover model list as a navigable column. Each row links to the
 * model's detail and highlights the active one.
 */
export { default as ModelsDiscoverList } from './ModelsDiscoverList.svelte';

/**
 * **ModelsDiscoverListSearch** - Sidebar search input
 *
 * Debounced search field for the model list.
 */
export { default as ModelsDiscoverListSearch } from './ModelsDiscoverListSearch.svelte';

/**
 * **ModelsDiscoverListItem** - Single sidebar row
 *
 * One model entry in the discover sidebar list, selectable via `onSelect`.
 */
export { default as ModelsDiscoverListItem } from './ModelsDiscoverListItem.svelte';

/**
 * **ModelsDiscoverListItemSkeleton** - Skeleton sidebar row
 *
 * Pulsing placeholder matching a ModelsDiscoverListItem row, shown while the
 * list is loading.
 */
export { default as ModelsDiscoverListItemSkeleton } from './ModelsDiscoverListItemSkeleton.svelte';
