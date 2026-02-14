/**
 *
 * FORMS & INPUTS
 *
 * Form-related utility components.
 *
 */

/**
 * **SearchInput** - Search field with clear button
 *
 * Input field optimized for search with clear button and keyboard handling.
 * Supports placeholder, autofocus, and change callbacks.
 */
export { default as SearchInput } from './SearchInput.svelte';

/**
 * **KeyValuePairs** - Editable key-value list
 *
 * Dynamic list of key-value pairs with add/remove functionality.
 * Used for HTTP headers, metadata, and configuration.
 *
 * **Features:**
 * - Add new pairs with button
 * - Remove individual pairs
 * - Customizable placeholders and labels
 * - Empty state message
 * - Auto-resize value textarea
 */
export { default as KeyValuePairs } from './KeyValuePairs.svelte';
