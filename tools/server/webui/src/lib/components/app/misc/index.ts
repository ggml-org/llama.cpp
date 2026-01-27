/**
 *
 * MISC
 *
 * Miscellaneous utility components.
 *
 */

/**
 * **TruncatedText** - Text with ellipsis and tooltip
 *
 * Displays text with automatic truncation and full content in tooltip.
 * Useful for long names or paths in constrained spaces.
 */
export { default as TruncatedText } from './TruncatedText.svelte';

/**
 * **ConversationSelection** - Multi-select conversation picker
 *
 * List of conversations with checkboxes for multi-selection.
 * Used in import/export dialogs for selecting conversations.
 *
 * **Features:**
 * - Search/filter conversations by name
 * - Select all / deselect all controls
 * - Shift-click for range selection
 * - Message count display per conversation
 * - Mode-specific UI (export vs import)
 */
export { default as ConversationSelection } from './ConversationSelection.svelte';
