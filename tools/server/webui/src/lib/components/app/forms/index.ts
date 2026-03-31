/**
 *
 * FORMS & INPUTS
 *
 * Form-related utility components.
 *
 */

/**
 * **InputWithSuggestions** - Input field with autocomplete suggestions
 *
 * Text input with dropdown suggestions and keyboard navigation.
 * Supports autocomplete functionality with suggestion loading.
 *
 * **Features:**
 * - Autocomplete dropdown with suggestions
 * - Keyboard navigation (arrow keys, enter)
 * - Loading state for suggestions
 * - Focus and blur handling
 */
export { default as InputWithSuggestions } from './InputWithSuggestions.svelte';

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

/**
 * **SearchInput** - Search field with clear button
 *
 * Input field optimized for search with clear button and keyboard handling.
 * Supports placeholder, autofocus, and change callbacks.
 */
export { default as SearchInput } from './SearchInput.svelte';
