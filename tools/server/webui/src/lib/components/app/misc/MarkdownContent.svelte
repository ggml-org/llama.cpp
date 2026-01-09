<script lang="ts">
	import { remark } from 'remark';
	import remarkBreaks from 'remark-breaks';
	import remarkGfm from 'remark-gfm';
	import remarkMath from 'remark-math';
	import rehypeHighlight from 'rehype-highlight';
	import remarkRehype from 'remark-rehype';
	import rehypeKatex from 'rehype-katex';
	import rehypeStringify from 'rehype-stringify';
	import type { Root as HastRoot, RootContent as HastRootContent } from 'hast';
	import type { Root as MdastRoot } from 'mdast';
	import { browser } from '$app/environment';
	import { onDestroy, tick } from 'svelte';
	import { rehypeRestoreTableHtml } from '$lib/markdown/table-html-restorer';
	import { rehypeEnhanceLinks } from '$lib/markdown/enhance-links';
	import { rehypeEnhanceCodeBlocks } from '$lib/markdown/enhance-code-blocks';
	import { remarkLiteralHtml } from '$lib/markdown/literal-html';
	import { copyCodeToClipboard, preprocessLaTeX } from '$lib/utils';
	import '$styles/katex-custom.scss';
	import githubDarkCss from 'highlight.js/styles/github-dark.css?inline';
	import githubLightCss from 'highlight.js/styles/github.css?inline';
	import { mode } from 'mode-watcher';
	import CodePreviewDialog from './CodePreviewDialog.svelte';

	interface Props {
		content: string;
		class?: string;
	}

	interface MarkdownBlock {
		id: string;
		html: string;
	}

	let { content, class: className = '' }: Props = $props();

	let containerRef = $state<HTMLDivElement>();
	let renderedBlocks = $state<MarkdownBlock[]>([]);
	let unstableBlockHtml = $state('');
	let previewDialogOpen = $state(false);
	let previewCode = $state('');
	let previewLanguage = $state('text');

	let pendingMarkdown: string | null = null;
	let isProcessing = false;

	const themeStyleId = `highlight-theme-${(window.idxThemeStyle = (window.idxThemeStyle ?? 0) + 1)}`;

	let processor = $derived(() => {
		return remark()
			.use(remarkGfm) // GitHub Flavored Markdown
			.use(remarkMath) // Parse $inline$ and $$block$$ math
			.use(remarkBreaks) // Convert line breaks to <br>
			.use(remarkLiteralHtml) // Treat raw HTML as literal text with preserved indentation
			.use(remarkRehype) // Convert Markdown AST to rehype
			.use(rehypeKatex) // Render math using KaTeX
			.use(rehypeHighlight) // Add syntax highlighting
			.use(rehypeRestoreTableHtml) // Restore limited HTML (e.g., <br>, <ul>) inside Markdown tables
			.use(rehypeEnhanceLinks) // Add target="_blank" to links
			.use(rehypeEnhanceCodeBlocks) // Wrap code blocks with header and actions
			.use(rehypeStringify, { allowDangerousHtml: true }); // Convert to HTML string
	});

	/**
	 * Removes click event listeners from copy and preview buttons.
	 * Called on component destroy.
	 */
	function cleanupEventListeners() {
		if (!containerRef) return;

		const copyButtons = containerRef.querySelectorAll<HTMLButtonElement>('.copy-code-btn');
		const previewButtons = containerRef.querySelectorAll<HTMLButtonElement>('.preview-code-btn');

		for (const button of copyButtons) {
			button.removeEventListener('click', handleCopyClick);
		}

		for (const button of previewButtons) {
			button.removeEventListener('click', handlePreviewClick);
		}
	}

	/**
	 * Removes this component's highlight.js theme style from the document head.
	 * Called on component destroy to clean up injected styles.
	 */
	function cleanupHighlightTheme() {
		if (!browser) return;

		const existingTheme = document.getElementById(themeStyleId);
		existingTheme?.remove();
	}

	/**
	 * Loads the appropriate highlight.js theme based on dark/light mode.
	 * Injects a scoped style element into the document head.
	 * @param isDark - Whether to load the dark theme (true) or light theme (false)
	 */
	function loadHighlightTheme(isDark: boolean) {
		if (!browser) return;

		const existingTheme = document.getElementById(themeStyleId);
		existingTheme?.remove();

		const style = document.createElement('style');
		style.id = themeStyleId;
		style.textContent = isDark ? githubDarkCss : githubLightCss;

		document.head.appendChild(style);
	}

	/**
	 * Extracts code information from a button click target within a code block.
	 * @param target - The clicked button element
	 * @returns Object with rawCode and language, or null if extraction fails
	 */
	function getCodeInfoFromTarget(target: HTMLElement) {
		const wrapper = target.closest('.code-block-wrapper');

		if (!wrapper) {
			console.error('No wrapper found');
			return null;
		}

		const codeElement = wrapper.querySelector<HTMLElement>('code[data-code-id]');

		if (!codeElement) {
			console.error('No code element found in wrapper');
			return null;
		}

		const rawCode = codeElement.textContent ?? '';

		const languageLabel = wrapper.querySelector<HTMLElement>('.code-language');
		const language = languageLabel?.textContent?.trim() || 'text';

		return { rawCode, language };
	}

	/**
	 * Generates a unique identifier for a HAST node based on its position.
	 * Used for stable block identification during incremental rendering.
	 * @param node - The HAST root content node
	 * @param indexFallback - Fallback index if position is unavailable
	 * @returns Unique string identifier for the node
	 */
	function getHastNodeId(node: HastRootContent, indexFallback: number): string {
		const position = node.position;

		if (position?.start?.offset != null && position?.end?.offset != null) {
			return `hast-${position.start.offset}-${position.end.offset}`;
		}

		return `${node.type}-${indexFallback}`;
	}

	/**
	 * Handles click events on copy buttons within code blocks.
	 * Copies the raw code content to the clipboard.
	 * @param event - The click event from the copy button
	 */
	async function handleCopyClick(event: Event) {
		event.preventDefault();
		event.stopPropagation();

		const target = event.currentTarget as HTMLButtonElement | null;

		if (!target) {
			return;
		}

		const info = getCodeInfoFromTarget(target);

		if (!info) {
			return;
		}

		try {
			await copyCodeToClipboard(info.rawCode);
		} catch (error) {
			console.error('Failed to copy code:', error);
		}
	}

	/**
	 * Handles preview dialog open state changes.
	 * Clears preview content when dialog is closed.
	 * @param open - Whether the dialog is being opened or closed
	 */
	function handlePreviewDialogOpenChange(open: boolean) {
		previewDialogOpen = open;

		if (!open) {
			previewCode = '';
			previewLanguage = 'text';
		}
	}

	/**
	 * Handles click events on preview buttons within HTML code blocks.
	 * Opens a preview dialog with the rendered HTML content.
	 * @param event - The click event from the preview button
	 */
	function handlePreviewClick(event: Event) {
		event.preventDefault();
		event.stopPropagation();

		const target = event.currentTarget as HTMLButtonElement | null;

		if (!target) {
			return;
		}

		const info = getCodeInfoFromTarget(target);

		if (!info) {
			return;
		}

		previewCode = info.rawCode;
		previewLanguage = info.language;
		previewDialogOpen = true;
	}

	/**
	 * Processes markdown content into stable and unstable HTML blocks.
	 * Uses incremental rendering: stable blocks are cached, unstable block is re-rendered.
	 * @param markdown - The raw markdown string to process
	 */
	async function processMarkdown(markdown: string) {
		if (!markdown) {
			renderedBlocks = [];
			unstableBlockHtml = '';
			return;
		}

		const normalized = preprocessLaTeX(markdown);
		const processorInstance = processor();
		const ast = processorInstance.parse(normalized) as MdastRoot;
		const processedRoot = (await processorInstance.run(ast)) as HastRoot;
		const processedChildren = processedRoot.children ?? [];
		const stableCount = Math.max(processedChildren.length - 1, 0);
		const nextBlocks: MarkdownBlock[] = [];

		for (let index = 0; index < stableCount; index++) {
			const hastChild = processedChildren[index];
			const id = getHastNodeId(hastChild, index);
			const existing = renderedBlocks[index];

			if (existing && existing.id === id) {
				nextBlocks.push(existing);
				continue;
			}

			const html = stringifyProcessedNode(
				processorInstance,
				processedRoot,
				processedChildren[index]
			);

			nextBlocks.push({ id, html });
		}

		let unstableHtml = '';

		if (processedChildren.length > stableCount) {
			const unstableChild = processedChildren[stableCount];
			unstableHtml = stringifyProcessedNode(processorInstance, processedRoot, unstableChild);
		}

		renderedBlocks = nextBlocks;
		await tick(); // Force DOM sync before updating unstable HTML block
		unstableBlockHtml = unstableHtml;
	}

	/**
	 * Attaches click event listeners to copy and preview buttons in code blocks.
	 * Uses data-listener-bound attribute to prevent duplicate bindings.
	 */
	function setupCodeBlockActions() {
		if (!containerRef) return;

		const wrappers = containerRef.querySelectorAll<HTMLElement>('.code-block-wrapper');

		for (const wrapper of wrappers) {
			const copyButton = wrapper.querySelector<HTMLButtonElement>('.copy-code-btn');
			const previewButton = wrapper.querySelector<HTMLButtonElement>('.preview-code-btn');

			if (copyButton && copyButton.dataset.listenerBound !== 'true') {
				copyButton.dataset.listenerBound = 'true';
				copyButton.addEventListener('click', handleCopyClick);
			}

			if (previewButton && previewButton.dataset.listenerBound !== 'true') {
				previewButton.dataset.listenerBound = 'true';
				previewButton.addEventListener('click', handlePreviewClick);
			}
		}
	}

	/**
	 * Converts a single HAST node to an enhanced HTML string.
	 * Applies link and code block enhancements to the output.
	 * @param processorInstance - The remark/rehype processor instance
	 * @param processedRoot - The full processed HAST root (for context)
	 * @param child - The specific HAST child node to stringify
	 * @returns Enhanced HTML string representation of the node
	 */
	function stringifyProcessedNode(
		processorInstance: ReturnType<typeof processor>,
		processedRoot: HastRoot,
		child: unknown
	) {
		const root: HastRoot = {
			...(processedRoot as HastRoot),
			children: [child as never]
		};

		return processorInstance.stringify(root);
	}

	/**
	 * Queues markdown for processing with coalescing support.
	 * Only processes the latest markdown when multiple updates arrive quickly.
	 * @param markdown - The markdown content to render
	 */
	async function updateRenderedBlocks(markdown: string) {
		pendingMarkdown = markdown;

		if (isProcessing) {
			return;
		}

		isProcessing = true;

		try {
			while (pendingMarkdown !== null) {
				const nextMarkdown = pendingMarkdown;
				pendingMarkdown = null;

				await processMarkdown(nextMarkdown);
			}
		} catch (error) {
			console.error('Failed to process markdown:', error);
			renderedBlocks = [];
			unstableBlockHtml = markdown.replace(/\n/g, '<br>');
		} finally {
			isProcessing = false;
		}
	}

	function escapeHtml(text: string): string {
		return text
			.replace(/&/g, '&amp;')
			.replace(/</g, '&lt;')
			.replace(/>/g, '&gt;')
			.replace(/\n/g, '<br>');
	}

	$effect(() => {
		const currentMode = mode.current;
		const isDark = currentMode === 'dark';

		loadHighlightTheme(isDark);
	});

	$effect(() => {
		updateRenderedBlocks(content);
	});

	$effect(() => {
		const hasRenderedBlocks = renderedBlocks.length > 0;
		const hasUnstableBlock = Boolean(unstableBlockHtml);

		if ((hasRenderedBlocks || hasUnstableBlock) && containerRef) {
			setupCodeBlockActions();
		}
	});

	onDestroy(() => {
		cleanupEventListeners();
		cleanupHighlightTheme();
	});
</script>

<div bind:this={containerRef} class={className}>
	{#each renderedBlocks as block (block.id)}
		<div class="markdown-block" data-block-id={block.id}>
			<!-- eslint-disable-next-line no-at-html-tags -->
			{@html block.html}
		</div>
	{/each}

	{#if unstableBlockHtml}
		<div class="markdown-block markdown-block--unstable" data-block-id="unstable">
			<!-- eslint-disable-next-line no-at-html-tags -->
			{@html unstableBlockHtml}
		</div>
	{/if}
</div>

<CodePreviewDialog
	open={previewDialogOpen}
	code={previewCode}
	language={previewLanguage}
	onOpenChange={handlePreviewDialogOpenChange}
/>

<style>
	.markdown-block,
	.markdown-block--unstable {
		display: contents;
	}

	/* Base typography styles */
	div :global(p:not(:last-child)) {
		margin-bottom: 1rem;
		line-height: 1.75;
	}

	div :global(:is(h1, h2, h3, h4, h5, h6):first-child) {
		margin-top: 0;
	}

	/* Headers with consistent spacing */
	div :global(h1) {
		font-size: 1.875rem;
		font-weight: 700;
		line-height: 1.2;
		margin: 1.5rem 0 0.75rem 0;
	}

	div :global(h2) {
		font-size: 1.5rem;
		font-weight: 600;
		line-height: 1.3;
		margin: 1.25rem 0 0.5rem 0;
	}

	div :global(h3) {
		font-size: 1.25rem;
		font-weight: 600;
		margin: 1.5rem 0 0.5rem 0;
		line-height: 1.4;
	}

	div :global(h4) {
		font-size: 1.125rem;
		font-weight: 600;
		margin: 0.75rem 0 0.25rem 0;
	}

	div :global(h5) {
		font-size: 1rem;
		font-weight: 600;
		margin: 0.5rem 0 0.25rem 0;
	}

	div :global(h6) {
		font-size: 0.875rem;
		font-weight: 600;
		margin: 0.5rem 0 0.25rem 0;
	}

	/* Text formatting */
	div :global(strong) {
		font-weight: 600;
	}

	div :global(em) {
		font-style: italic;
	}

	div :global(del) {
		text-decoration: line-through;
		opacity: 0.7;
	}

	/* Inline code */
	div :global(code:not(pre code)) {
		background: var(--muted);
		color: var(--muted-foreground);
		padding: 0.125rem 0.375rem;
		border-radius: 0.375rem;
		font-size: 0.875rem;
		font-family:
			ui-monospace, SFMono-Regular, 'SF Mono', Monaco, 'Cascadia Code', 'Roboto Mono', Consolas,
			'Liberation Mono', Menlo, monospace;
	}

	/* Links */
	div :global(a) {
		color: var(--primary);
		text-decoration: underline;
		text-underline-offset: 2px;
		transition: color 0.2s ease;
	}

	div :global(a:hover) {
		color: var(--primary);
	}

	/* Lists */
	div :global(ul) {
		list-style-type: disc;
		margin-left: 1.5rem;
		margin-bottom: 1rem;
	}

	div :global(ol) {
		list-style-type: decimal;
		margin-left: 1.5rem;
		margin-bottom: 1rem;
	}

	div :global(li) {
		margin-bottom: 0.25rem;
		padding-left: 0.5rem;
	}

	div :global(li::marker) {
		color: var(--muted-foreground);
	}

	/* Nested lists */
	div :global(ul ul) {
		list-style-type: circle;
		margin-top: 0.25rem;
		margin-bottom: 0.25rem;
	}

	div :global(ol ol) {
		list-style-type: lower-alpha;
		margin-top: 0.25rem;
		margin-bottom: 0.25rem;
	}

	/* Task lists */
	div :global(.task-list-item) {
		list-style: none;
		margin-left: 0;
		padding-left: 0;
	}

	div :global(.task-list-item-checkbox) {
		margin-right: 0.5rem;
		margin-top: 0.125rem;
	}

	/* Blockquotes */
	div :global(blockquote) {
		border-left: 4px solid var(--border);
		padding: 0.5rem 1rem;
		margin: 1.5rem 0;
		font-style: italic;
		color: var(--muted-foreground);
		background: var(--muted);
		border-radius: 0 0.375rem 0.375rem 0;
	}

	/* Tables */
	div :global(table) {
		width: 100%;
		margin: 1.5rem 0;
		border-collapse: collapse;
		border: 1px solid var(--border);
		border-radius: 0.375rem;
		overflow: hidden;
	}

	div :global(th) {
		background: hsl(var(--muted) / 0.3);
		border: 1px solid var(--border);
		padding: 0.5rem 0.75rem;
		text-align: left;
		font-weight: 600;
	}

	div :global(td) {
		border: 1px solid var(--border);
		padding: 0.5rem 0.75rem;
	}

	div :global(tr:nth-child(even)) {
		background: hsl(var(--muted) / 0.1);
	}

	/* User message markdown should keep table borders visible on light primary backgrounds */
	div.markdown-user-content :global(table),
	div.markdown-user-content :global(th),
	div.markdown-user-content :global(td),
	div.markdown-user-content :global(.table-wrapper) {
		border-color: currentColor;
	}

	/* Horizontal rules */
	div :global(hr) {
		border: none;
		border-top: 1px solid var(--border);
		margin: 1.5rem 0;
	}

	/* Images */
	div :global(img) {
		border-radius: 0.5rem;
		box-shadow:
			0 1px 3px 0 rgb(0 0 0 / 0.1),
			0 1px 2px -1px rgb(0 0 0 / 0.1);
		margin: 1.5rem 0;
		max-width: 100%;
		height: auto;
	}

	/* Code blocks */

	div :global(.code-block-wrapper) {
		margin: 1.5rem 0;
		border-radius: 0.75rem;
		overflow: hidden;
		border: 1px solid var(--border);
		background: var(--code-background);
	}

	div :global(.code-block-header) {
		display: flex;
		justify-content: space-between;
		align-items: center;
		padding: 0.5rem 1rem;
		background: hsl(var(--muted) / 0.5);
		border-bottom: 1px solid var(--border);
		font-size: 0.875rem;
	}

	div :global(.code-language) {
		color: var(--code-foreground);
		font-weight: 500;
		font-family:
			ui-monospace, SFMono-Regular, 'SF Mono', Monaco, 'Cascadia Code', 'Roboto Mono', Consolas,
			'Liberation Mono', Menlo, monospace;
		text-transform: uppercase;
		font-size: 0.75rem;
		letter-spacing: 0.05em;
	}

	div :global(.code-block-actions) {
		display: flex;
		align-items: center;
		gap: 0.5rem;
	}

	div :global(.copy-code-btn),
	div :global(.preview-code-btn) {
		display: flex;
		align-items: center;
		justify-content: center;
		padding: 0;
		background: transparent;
		color: var(--code-foreground);
		cursor: pointer;
		transition: all 0.2s ease;
	}

	div :global(.copy-code-btn:hover),
	div :global(.preview-code-btn:hover) {
		transform: scale(1.05);
	}

	div :global(.copy-code-btn:active),
	div :global(.preview-code-btn:active) {
		transform: scale(0.95);
	}

	div :global(.code-block-wrapper pre) {
		background: transparent;
		padding: 1rem;
		margin: 0;
		overflow-x: auto;
		border-radius: 0;
		border: none;
		font-size: 0.875rem;
		line-height: 1.5;
	}

	div :global(pre) {
		background: var(--muted);
		margin: 1.5rem 0;
		overflow-x: auto;
		border-radius: 1rem;
		border: none;
	}

	div :global(code) {
		background: transparent;
		color: var(--code-foreground);
	}

	/* Mentions and hashtags */
	div :global(.mention) {
		color: hsl(var(--primary));
		font-weight: 500;
		text-decoration: none;
	}

	div :global(.mention:hover) {
		text-decoration: underline;
	}

	div :global(.hashtag) {
		color: hsl(var(--primary));
		font-weight: 500;
		text-decoration: none;
	}

	div :global(.hashtag:hover) {
		text-decoration: underline;
	}

	/* Advanced table enhancements */
	div :global(table) {
		transition: all 0.2s ease;
	}

	div :global(table:hover) {
		box-shadow:
			0 4px 6px -1px rgb(0 0 0 / 0.1),
			0 2px 4px -2px rgb(0 0 0 / 0.1);
	}

	div :global(th:hover),
	div :global(td:hover) {
		background: var(--muted);
	}

	/* Disable hover effects when rendering user messages */
	.markdown-user-content :global(a),
	.markdown-user-content :global(a:hover) {
		color: var(--primary-foreground);
	}

	.markdown-user-content :global(table:hover) {
		box-shadow: none;
	}

	.markdown-user-content :global(th:hover),
	.markdown-user-content :global(td:hover) {
		background: inherit;
	}

	/* Enhanced blockquotes */
	div :global(blockquote) {
		transition: all 0.2s ease;
		position: relative;
	}

	div :global(blockquote:hover) {
		border-left-width: 6px;
		background: var(--muted);
		transform: translateX(2px);
	}

	div :global(blockquote::before) {
		content: '"';
		position: absolute;
		top: -0.5rem;
		left: 0.5rem;
		font-size: 3rem;
		color: var(--muted-foreground);
		font-family: serif;
		line-height: 1;
	}

	/* Enhanced images */
	div :global(img) {
		transition: all 0.3s ease;
		cursor: pointer;
	}

	div :global(img:hover) {
		transform: scale(1.02);
		box-shadow:
			0 10px 15px -3px rgb(0 0 0 / 0.1),
			0 4px 6px -4px rgb(0 0 0 / 0.1);
	}

	/* Image zoom overlay */
	div :global(.image-zoom-overlay) {
		position: fixed;
		top: 0;
		left: 0;
		right: 0;
		bottom: 0;
		background: rgba(0, 0, 0, 0.8);
		display: flex;
		align-items: center;
		justify-content: center;
		z-index: 1000;
		cursor: pointer;
	}

	div :global(.image-zoom-overlay img) {
		max-width: 90vw;
		max-height: 90vh;
		border-radius: 0.5rem;
		box-shadow: 0 25px 50px -12px rgb(0 0 0 / 0.25);
	}

	/* Enhanced horizontal rules */
	div :global(hr) {
		border: none;
		height: 2px;
		background: linear-gradient(to right, transparent, var(--border), transparent);
		margin: 2rem 0;
		position: relative;
	}

	div :global(hr::after) {
		content: '';
		position: absolute;
		top: 50%;
		left: 50%;
		transform: translate(-50%, -50%);
		width: 1rem;
		height: 1rem;
		background: var(--border);
		border-radius: 50%;
	}

	/* Scrollable tables */
	div :global(.table-wrapper) {
		overflow-x: auto;
		margin: 1.5rem 0;
		border-radius: 0.5rem;
		border: 1px solid var(--border);
	}

	div :global(.table-wrapper table) {
		margin: 0;
		border: none;
	}

	/* Responsive adjustments */
	@media (max-width: 640px) {
		div :global(h1) {
			font-size: 1.5rem;
		}

		div :global(h2) {
			font-size: 1.25rem;
		}

		div :global(h3) {
			font-size: 1.125rem;
		}

		div :global(table) {
			font-size: 0.875rem;
		}

		div :global(th),
		div :global(td) {
			padding: 0.375rem 0.5rem;
		}

		div :global(.table-wrapper) {
			margin: 0.5rem -1rem;
			border-radius: 0;
			border-left: none;
			border-right: none;
		}
	}

	/* Dark mode adjustments */
	@media (prefers-color-scheme: dark) {
		div :global(blockquote:hover) {
			background: var(--muted);
		}
	}
</style>
