/**
 *
 * MISC
 *
 * Reusable utility components used across the application.
 * Includes content rendering, UI primitives, and helper components.
 *
 */

/**
 *
 * CONTENT RENDERING
 *
 * Components for rendering rich content: markdown, code, and previews.
 *
 */

/**
 * **MarkdownContent** - Rich markdown renderer
 *
 * Renders markdown content with syntax highlighting, LaTeX math,
 * tables, links, and code blocks. Optimized for streaming with
 * incremental block-based rendering.
 *
 * **Features:**
 * - GFM (GitHub Flavored Markdown): tables, task lists, strikethrough
 * - LaTeX math via KaTeX (`$inline$` and `$$block$$`)
 * - Syntax highlighting (highlight.js) with language detection
 * - Code copy buttons with click feedback
 * - External links open in new tab with security attrs
 * - Image attachment resolution from message extras
 * - Dark/light theme support (auto-switching)
 * - Streaming-optimized incremental rendering
 * - Code preview dialog for large blocks
 *
 * @example
 * ```svelte
 * <MarkdownContent content={message.content} {message} />
 * ```
 */
export { default as MarkdownContent } from './MarkdownContent.svelte';

/**
 * **SyntaxHighlightedCode** - Code syntax highlighting
 *
 * Renders code with syntax highlighting using highlight.js.
 * Supports theme switching and scrollable containers.
 *
 * **Features:**
 * - Auto language detection with fallback
 * - Dark/light theme auto-switching
 * - Scrollable container with configurable max dimensions
 * - Monospace font styling
 * - Preserves whitespace and formatting
 *
 * @example
 * ```svelte
 * <SyntaxHighlightedCode code={jsonString} language="json" />
 * ```
 */
export { default as SyntaxHighlightedCode } from './SyntaxHighlightedCode.svelte';

/**
 * **CodePreviewDialog** - Full-screen code preview
 *
 * Full-screen dialog for previewing HTML/code in an isolated iframe.
 * Used by MarkdownContent for previewing rendered HTML blocks.
 *
 * **Features:**
 * - Full viewport iframe preview
 * - Sandboxed execution (allow-scripts only)
 * - Close button with mix-blend-difference for visibility
 * - Clears content when closed for security
 */
export { default as CodePreviewDialog } from './CodePreviewDialog.svelte';

/**
 *
 * COLLAPSIBLE & EXPANDABLE
 *
 * Components for showing/hiding content sections.
 *
 */

/**
 * **CollapsibleContentBlock** - Expandable content card
 *
 * Reusable collapsible card with header, icon, and auto-scroll.
 * Used for tool calls and reasoning blocks in chat messages.
 *
 * **Features:**
 * - Collapsible content with smooth animation
 * - Custom icon and title display
 * - Optional subtitle/status text
 * - Auto-scroll during streaming (pauses on user scroll)
 * - Configurable max height with overflow scroll
 *
 * @example
 * ```svelte
 * <CollapsibleContentBlock
 *   bind:open
 *   icon={BrainIcon}
 *   title="Thinking..."
 *   isStreaming={true}
 * >
 *   {reasoningContent}
 * </CollapsibleContentBlock>
 * ```
 */
export { default as CollapsibleContentBlock } from './CollapsibleContentBlock.svelte';

/**
 * **TruncatedText** - Text with ellipsis and tooltip
 *
 * Displays text with automatic truncation and full content in tooltip.
 * Useful for long names or paths in constrained spaces.
 */
export { default as TruncatedText } from './TruncatedText.svelte';

/**
 *
 * DROPDOWNS & MENUS
 *
 * Components for dropdown menus and action selection.
 *
 */

/**
 * **DropdownMenuSearchable** - Filterable dropdown menu
 *
 * Generic dropdown with search input for filtering options.
 * Uses Svelte snippets for flexible content rendering.
 *
 * **Features:**
 * - Search/filter input with clear on close
 * - Keyboard navigation support
 * - Custom trigger, content, and footer via snippets
 * - Empty state message
 * - Disabled state
 * - Configurable alignment and width
 *
 * @example
 * ```svelte
 * <DropdownMenuSearchable
 *   bind:open
 *   bind:searchValue
 *   placeholder="Search..."
 *   isEmpty={filteredItems.length === 0}
 * >
 *   {#snippet trigger()}<Button>Select</Button>{/snippet}
 *   {#snippet children()}{#each items as item}<Item {item} />{/each}{/snippet}
 * </DropdownMenuSearchable>
 * ```
 */
export { default as DropdownMenuSearchable } from './DropdownMenuSearchable.svelte';

/**
 * **DropdownMenuActions** - Multi-action dropdown menu
 *
 * Dropdown menu for multiple action options with icons and shortcuts.
 * Supports destructive variants and keyboard shortcut hints.
 *
 * **Features:**
 * - Configurable trigger icon with tooltip
 * - Action items with icons and labels
 * - Destructive variant styling
 * - Keyboard shortcut display
 * - Separator support between groups
 *
 * @example
 * ```svelte
 * <DropdownMenuActions
 *   triggerIcon={MoreHorizontal}
 *   triggerTooltip="More actions"
 *   actions={[
 *     { icon: Edit, label: 'Edit', onclick: handleEdit },
 *     { icon: Trash, label: 'Delete', onclick: handleDelete, variant: 'destructive' }
 *   ]}
 * />
 * ```
 */
export { default as DropdownMenuActions } from './DropdownMenuActions.svelte';

/**
 *
 * BUTTONS & ACTIONS
 *
 * Small interactive components for user actions.
 *
 */

/** Styled button for action triggers with icon support. */
export { default as ActionButton } from './ActionButton.svelte';

/** Copy-to-clipboard button with success feedback. */
export { default as CopyToClipboardIcon } from './CopyToClipboardIcon.svelte';

/** Remove/delete button with X icon. */
export { default as RemoveButton } from './RemoveButton.svelte';

/**
 *
 * BADGES & INDICATORS
 *
 * Small visual indicators for status and metadata.
 *
 */

/** Badge displaying chat statistics (tokens, timing). */
export { default as BadgeChatStatistic } from './BadgeChatStatistic.svelte';

/** Generic info badge with optional tooltip and click handler. */
export { default as BadgeInfo } from './BadgeInfo.svelte';

/** Badge indicating model modality (vision, audio, tools). */
export { default as BadgeModality } from './BadgeModality.svelte';

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
 *
 * KEYBOARD & SHORTCUTS
 *
 * Components for displaying keyboard shortcuts.
 *
 */

/** Display for keyboard shortcut hints (e.g., "⌘ + Enter"). */
export { default as KeyboardShortcutInfo } from './KeyboardShortcutInfo.svelte';
