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
 * **MarkdownEditor** - Live-preview markdown editor (CodeMirror)
 *
 * Edits markdown source in place: every block the cursor is not in renders
 * exactly like MarkdownContent, and the block being edited shows its raw text
 * in the same typography. The document is never rewritten, so the value stays
 * byte-exact. Used by the chat form input.
 *
 * @example
 * ```svelte
 * <MarkdownEditor value={text} onChange={(v) => (text = v)} placeholder="Write..." />
 * ```
 */
export { default as MarkdownEditor } from './MarkdownEditor/MarkdownEditor.svelte';

/**
 * **MentionText** - Plain text with file mention badges
 *
 * Renders a message verbatim, turning only `[name](file://path)` links
 * into the same badge chips the markdown path draws. Nothing else is
 * interpreted, so pasted code keeps its `#` comments and underscores.
 *
 * @example
 * ```svelte
 * <span class="whitespace-pre-wrap"><MentionText content={message.content} /></span>
 * ```
 */
export { default as MentionText } from './MentionText.svelte';

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
export { SyntaxHighlightedCode } from '@ggml-org/llama-ui/app/content';

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
 * **MermaidPreview** - Pan and zoom viewer for a diagram's svg
 *
 * Shows a diagram (a Pele-drawn Mermaid block or an svg block) fitted to its box, with
 * zoom, pan and download controls.
 *
 * **Features:**
 * - Fits the diagram until the user pans or zooms, also as the box resizes
 * - Wheel and trackpad pinch zoom around the pointer; zoom buttons and +/- keys zoom
 *   around the middle of the box, wherever the diagram has been panned to
 * - Drag to pan, two-finger pinch on touch, double-click to zoom in, arrow keys to pan
 * - Reset view (0) fits the diagram again
 * - Download as SVG with the theme colors written in
 *
 * @example
 * ```svelte
 * <MermaidPreview svgHtml={diagramSvg} />
 * ```
 */
export { default as MermaidPreview } from './MermaidPreview.svelte';
