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
export { default as MarkdownContent } from './MarkdownContent/MarkdownContent.svelte';

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
 *   title="Thinking..."
 *   isStreaming
 * >
 *   {reasoningContent}
 * </CollapsibleContentBlock>
 * ```
 */
export { default as CollapsibleContentBlock } from './CollapsibleContentBlock.svelte';

/**
 * **CollapsibleTerminalBlock** - Expandable content card with a terminal-style frame
 *
 * Same shape as CollapsibleContentBlock, but with a `code-background`
 * fill, subtle border, and tightened padding suited for shell command
 * output and similar dense / monospace content.
 *
 * @example
 * ```svelte
 * <CollapsibleTerminalBlock bind:open title="Run command">
 *   <pre>{output}</pre>
 * </CollapsibleTerminalBlock>
 * ```
 */
export { default as CollapsibleTerminalBlock } from './CollapsibleTerminalBlock.svelte';

/**
 * **MermaidPreview** - Interactive Mermaid diagram viewer
 *
 * Renders Mermaid-generated SVG diagrams with zoom, pan, and fit-to-view controls.
 *
 * **Features:**
 * - Mouse wheel zoom in/out
 * - Click-drag panning with pointer capture
 * - Fit to view and reset view controls
 * - Download as SVG
 * - Responsive scaling with viewBox detection
 *
 * @example
 * ```svelte
 * <MermaidPreview svgHtml={diagramSvg} />
 * ```
 */
export { default as MermaidPreview } from './MermaidPreview.svelte';
