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
 * <MarkdownContent content={message.content} attachments={message.extra} />
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
