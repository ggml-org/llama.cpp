/**
 *
 * MISC
 *
 * Miscellaneous utility components.
 *
 */

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

/**
 * Horizontal scrollable carousel with navigation arrows.
 * Used for displaying items in a horizontally scrollable container
 * with left/right navigation buttons that appear on hover.
 */
export { default as HorizontalScrollCarousel } from './HorizontalScrollCarousel.svelte';

/**
 * **TruncatedText** - Text with ellipsis and tooltip
 *
 * Displays text with automatic truncation and full content in tooltip.
 * Useful for long names or paths in constrained spaces.
 */
export { default as TruncatedText } from './TruncatedText.svelte';

/**
 * **KeyboardShortcutInfo** - Keyboard shortcut hint display
 *
 * Displays keyboard shortcut hints (e.g., "⌘ + Enter").
 * Supports special keys like shift, cmd, and custom text.
 */
export { default as KeyboardShortcutInfo } from './KeyboardShortcutInfo.svelte';
