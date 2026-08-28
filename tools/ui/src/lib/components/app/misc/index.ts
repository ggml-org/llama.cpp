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

/**
 * **ScrollCarousel** - Feature/carousel with center-aligned overflow controls
 *
 * Horizontal scrollable container with arrows that center the focused item.
 */
export { default as ScrollCarousel } from './ScrollCarousel.svelte';

/**
 * **CodeBlockActions** - Actions bar for code blocks (copy, preview)
 *
 * Displays copy-to-clipboard and preview buttons for code blocks.
 * Preview button is shown only for HTML code blocks.
 */
export { default as CodeBlockActions } from './CodeBlockActions.svelte';

/**
 * **Logo** - Application brand mark
 *
 * Inline SVG of the application logo. Accepts styling via the standard
 * `class` and `style` props and inherits color via `currentColor`.
 */
export { default as Logo } from './Logo.svelte';
