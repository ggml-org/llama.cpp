<script lang="ts">
	import { remark } from 'remark';
	import remarkBreaks from 'remark-breaks';
	import remarkGfm from 'remark-gfm';
	import remarkMath from 'remark-math';
	import rehypeHighlight from 'rehype-highlight';
	import { all as lowlightAll } from 'lowlight';
	import remarkRehype from 'remark-rehype';
	import rehypeKatex from 'rehype-katex';
	import rehypeStringify from 'rehype-stringify';
	import type { Root as HastRoot, RootContent as HastRootContent } from 'hast';
	import type { Root as MdastRoot } from 'mdast';
	import { browser } from '$app/environment';
	import { onDestroy, tick } from 'svelte';
	import { SvelteMap } from 'svelte/reactivity';
	import { rehypeRestoreTableHtml } from './plugins/rehype/table-html-restorer';
	import { rehypeEnhanceLinks } from './plugins/rehype/enhance-links';
	import { rehypeEnhanceCodeBlocks } from './plugins/rehype/enhance-code-blocks';
	import { rehypeEnhanceMermaidBlocks } from './plugins/rehype/enhance-mermaid-blocks';
	import { rehypeMermaidPre } from './plugins/rehype/mermaid-pre';
	import { rehypeResolveAttachmentImages } from './plugins/rehype/resolve-attachment-images';
	import { rehypeRtlSupport } from './plugins/rehype/rehype-rtl-support';
	import { remarkLiteralHtml } from './plugins/remark/literal-html';
	import {
		getHastNodeId,
		getMdastNodeHash,
		isAppendMode,
		getCodeInfoFromTarget
	} from './markdown-utils';
	import {
		preprocessLaTeX,
		getImageErrorFallbackHtml,
		copyCodeToClipboard,
		copyToClipboard
	} from '$lib/utils';
	import {
		IMAGE_NOT_ERROR_BOUND_SELECTOR,
		DATA_ERROR_BOUND_ATTR,
		DATA_ERROR_HANDLED_ATTR,
		BOOL_TRUE_STRING,
		SETTINGS_KEYS
	} from '$lib/constants';
	import { ColorMode, UrlProtocol } from '$lib/enums';
	import { FileTypeText } from '$lib/enums/files.enums';
	import { highlightCode, detectIncompleteCodeBlock, type IncompleteCodeBlock } from '$lib/utils';
	import '$styles/katex-custom.scss';
	import githubDarkCss from 'highlight.js/styles/github-dark.css?inline';
	import githubLightCss from 'highlight.js/styles/github.css?inline';
	import { mode } from 'mode-watcher';
	import {
		CodeBlockActions,
		DialogCodePreview,
		DialogMermaidPreview,
		ActionIconCopyToClipboard
	} from '$lib/components/app';
	import { createAutoScrollController } from '$lib/hooks/use-auto-scroll.svelte';
	import type { DatabaseMessageExtra } from '$lib/types/database';
	import { config } from '$lib/stores/settings.svelte';
	import { fadeInView } from '$lib/actions/fade-in-view.svelte';

	interface Props {
		attachments?: DatabaseMessageExtra[];
		content: string;
		class?: string;
		disableMath?: boolean;
	}

	interface MarkdownBlock {
		id: string;
		html: string;
		contentHash?: string;
	}

	let { content, attachments, class: className = '', disableMath = false }: Props = $props();

	let containerRef = $state<HTMLDivElement>();
	let renderedBlocks = $state<MarkdownBlock[]>([]);
	let unstableBlockHtml = $state('');
	let incompleteCodeBlock = $state<IncompleteCodeBlock | null>(null);
	let previewDialogOpen = $state(false);
	let previewCode = $state('');
	let previewLanguage = $state('text');
	let mermaidPreviewOpen = $state(false);
	let mermaidPreviewSvgHtml = $state('');

	let streamingCodeScrollContainer = $state<HTMLDivElement>();

	// Auto-scroll controller for streaming code block content
	const streamingAutoScroll = createAutoScrollController();

	let pendingMarkdown: string | null = null;
	let isProcessing = false;

	// Per-instance transform cache, avoids re-transforming stable blocks during streaming
	// Garbage collected when component is destroyed (on conversation change)
	const transformCache = new SvelteMap<string, string>();
	let previousContent = '';

	const themeStyleId = `highlight-theme-${(window.idxThemeStyle = (window.idxThemeStyle ?? 0) + 1)}`;

	let processor = $derived(() => {
		void attachments;
		// eslint-disable-next-line @typescript-eslint/no-explicit-any
		let proc: any = remark().use(remarkGfm); // GitHub Flavored Markdown

		if (!disableMath) {
			proc = proc.use(remarkMath); // Parse $inline$ and $$block$$ math
		}

		proc = proc
			.use(remarkBreaks) // Convert line breaks to <br>
			.use(remarkLiteralHtml) // Treat raw HTML as literal text with preserved indentation
			.use(remarkRehype); // Convert Markdown AST to rehype

		if (!disableMath) {
			proc = proc.use(rehypeKatex); // Render math using KaTeX
		}

		return proc
			.use(rehypeHighlight, {
				languages: lowlightAll,
				aliases: { [FileTypeText.XML]: [FileTypeText.SVELTE, FileTypeText.VUE] }
			}) // Add syntax highlighting
			.use(rehypeRestoreTableHtml) // Restore limited HTML (e.g., <br>, <ul>) inside Markdown tables
			.use(rehypeEnhanceLinks) // Add target="_blank" to links
			.use(rehypeMermaidPre) // Convert mermaid blocks to <pre class="mermaid">
			.use(rehypeEnhanceCodeBlocks) // Wrap code blocks with header and actions
			.use(rehypeEnhanceMermaidBlocks) // Wrap mermaid blocks with header and actions
			.use(rehypeResolveAttachmentImages, { attachments })
			.use(rehypeRtlSupport) // Add bidirectional text support
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
	 * Transforms a single MDAST node to HTML string with caching.

	/**
	 * Transforms a single MDAST node to HTML string with caching.
	 * Runs the full remark/rehype plugin pipeline (GFM, math, syntax highlighting, etc.)
	 * on an isolated single-node tree, then stringifies the resulting HAST to HTML.
	 * Results are cached by node position hash for streaming performance.
	 * @param processorInstance - The remark/rehype processor instance
	 * @param node - The MDAST node to transform
	 * @param index - Node index for hash fallback
	 * @returns Object containing the HTML string and cache hash
	 */
	async function transformMdastNode(
		processorInstance: ReturnType<typeof processor>,
		node: unknown,
		index: number
	): Promise<{ html: string; hash: string }> {
		const hash = getMdastNodeHash(node, index);

		const cached = transformCache.get(hash);
		if (cached) {
			return { html: cached, hash };
		}

		const singleNodeRoot = { type: 'root', children: [node] };
		const transformedRoot = (await processorInstance.run(singleNodeRoot as MdastRoot)) as HastRoot;
		const html = processorInstance.stringify(transformedRoot);

		transformCache.set(hash, html);

		return { html, hash };
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
	 * Incomplete code blocks are rendered using SyntaxHighlightedCode to maintain interactivity.
	 * @param markdown - The raw markdown string to process
	 */
	async function processMarkdown(markdown: string) {
		// Early exit if content unchanged (can happen with rapid coalescing)
		if (markdown === previousContent) {
			return;
		}

		if (!markdown) {
			renderedBlocks = [];
			unstableBlockHtml = '';
			incompleteCodeBlock = null;
			previousContent = '';
			return;
		}

		// Check for incomplete code block at the end of content
		const incompleteBlock = detectIncompleteCodeBlock(markdown);

		if (incompleteBlock) {
			// Process only the prefix (content before the incomplete code block)
			const prefixMarkdown = markdown.slice(0, incompleteBlock.openingIndex);

			if (prefixMarkdown.trim()) {
				const normalizedPrefix = preprocessLaTeX(prefixMarkdown);
				const processorInstance = processor();
				const ast = processorInstance.parse(normalizedPrefix) as MdastRoot;
				const mdastChildren = (ast as { children?: unknown[] }).children ?? [];
				const nextBlocks: MarkdownBlock[] = [];

				// Check if we're in append mode for cache reuse
				const appendMode = isAppendMode(prefixMarkdown, previousContent);
				const previousBlockCount = appendMode ? renderedBlocks.length : 0;

				// All prefix blocks are now stable since code block is separate
				for (let index = 0; index < mdastChildren.length; index++) {
					const child = mdastChildren[index];

					// In append mode, reuse previous blocks if unchanged
					if (appendMode && index < previousBlockCount) {
						const prevBlock = renderedBlocks[index];
						const currentHash = getMdastNodeHash(child, index);

						if (prevBlock?.contentHash === currentHash) {
							nextBlocks.push(prevBlock);

							continue;
						}
					}

					// Transform this block (with caching)
					const { html, hash } = await transformMdastNode(processorInstance, child, index);
					const id = getHastNodeId(
						{ position: (child as { position?: unknown }).position } as HastRootContent,
						index
					);

					nextBlocks.push({ id, html, contentHash: hash });
				}

				renderedBlocks = nextBlocks;
			} else {
				renderedBlocks = [];
			}

			previousContent = prefixMarkdown;
			unstableBlockHtml = '';
			incompleteCodeBlock = incompleteBlock;

			return;
		}

		// No incomplete code block - use standard processing
		incompleteCodeBlock = null;

		const normalized = preprocessLaTeX(markdown);
		const processorInstance = processor();
		const ast = processorInstance.parse(normalized) as MdastRoot;
		const mdastChildren = (ast as { children?: unknown[] }).children ?? [];
		const stableCount = Math.max(mdastChildren.length - 1, 0);
		const nextBlocks: MarkdownBlock[] = [];

		// Check if we're in append mode for cache reuse
		const appendMode = isAppendMode(markdown, previousContent);
		const previousBlockCount = appendMode ? renderedBlocks.length : 0;

		for (let index = 0; index < stableCount; index++) {
			const child = mdastChildren[index];

			// In append mode, reuse previous blocks if unchanged
			if (appendMode && index < previousBlockCount) {
				const prevBlock = renderedBlocks[index];
				const currentHash = getMdastNodeHash(child, index);
				if (prevBlock?.contentHash === currentHash) {
					nextBlocks.push(prevBlock);

					continue;
				}
			}

			// Transform this block (with caching)
			const { html, hash } = await transformMdastNode(processorInstance, child, index);
			const id = getHastNodeId(
				{ position: (child as { position?: unknown }).position } as HastRootContent,
				index
			);

			nextBlocks.push({ id, html, contentHash: hash });
		}

		let unstableHtml = '';

		if (mdastChildren.length > stableCount) {
			const unstableChild = mdastChildren[stableCount];
			const singleNodeRoot = { type: 'root', children: [unstableChild] };
			const transformedRoot = (await processorInstance.run(
				singleNodeRoot as MdastRoot
			)) as HastRoot;

			unstableHtml = processorInstance.stringify(transformedRoot);
		}

		renderedBlocks = nextBlocks;
		previousContent = markdown;
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
	 * Attaches error handlers to images to show fallback UI when loading fails (e.g., CORS).
	 * Uses data-error-bound attribute to prevent duplicate bindings.
	 */
	function setupImageErrorHandlers() {
		if (!containerRef) return;

		const images = containerRef.querySelectorAll<HTMLImageElement>(IMAGE_NOT_ERROR_BOUND_SELECTOR);

		for (const img of images) {
			img.dataset[DATA_ERROR_BOUND_ATTR] = BOOL_TRUE_STRING;
			img.addEventListener('error', handleImageError);
		}
	}

	/**
	 * Opens the mermaid diagram in a full-screen preview dialog with zoom/pan support.
	 * Also handles copy and preview button clicks for mermaid blocks.
	 * Uses event delegation: a single handler on the container.
	 */
	async function handleMermaidClick(event: MouseEvent) {
		const target = event.target as HTMLElement;

		// Check if clicking on copy or preview button in mermaid block
		const copyBtn = target.closest('.mermaid-block-wrapper .copy-code-btn');
		const previewBtn = target.closest('.mermaid-block-wrapper .preview-code-btn');

		if (copyBtn || previewBtn) {
			const wrapper = target.closest('.mermaid-block-wrapper');
			if (!wrapper) return;

			const preElement = wrapper.querySelector<HTMLElement>('pre.mermaid[data-mermaid-syntax]');
			if (!preElement) return;

			const mermaidSyntax = preElement.dataset.mermaidSyntax ?? '';

			if (copyBtn) {
				event.preventDefault();
				event.stopPropagation();
				try {
					await copyToClipboard(mermaidSyntax);
				} catch (error) {
					console.error('Failed to copy mermaid syntax:', error);
				}
				return;
			}

			if (previewBtn) {
				event.preventDefault();
				event.stopPropagation();
				const svg = preElement.querySelector('svg');
				if (!svg) return;
				mermaidPreviewSvgHtml = svg.outerHTML;
				mermaidPreviewOpen = true;
				return;
			}
		}

		// Otherwise, open preview when clicking on the mermaid diagram itself
		const mermaidEl = target.closest('.mermaid');
		if (!mermaidEl) return;

		const svg = mermaidEl.querySelector('svg');
		if (!svg) return;

		mermaidPreviewSvgHtml = svg.outerHTML;
		mermaidPreviewOpen = true;
	}

	/**
	 * Handles mermaid preview dialog open state changes.
	 * Cleans up SVG content when dialog is closed.
	 */
	function handleMermaidPreviewOpenChange(open: boolean) {
		mermaidPreviewOpen = open;
		if (!open) {
			mermaidPreviewSvgHtml = '';
		}
	}

	/**
	 * Renders mermaid diagrams that haven't been rendered yet.
	 * Called after each markdown content update.
	 * Marks nodes immediately to prevent duplicate renders during streaming.
	 * Reads mode.current before await to ensure reactive tracking.
	 */
	async function renderMermaidDiagrams() {
		if (!containerRef) return;

		const nodes = containerRef.querySelectorAll('pre.mermaid:not([data-mermaid-rendered])');
		if (nodes.length === 0) return;

		// Mark nodes immediately to prevent duplicate renders if called again during streaming.
		// This avoids needing a guard that would block node discovery.
		nodes.forEach((node) => node.setAttribute('data-mermaid-rendered', 'true'));

		// Read mode before await so Svelte tracks it reactively.
		const isDark = mode.current === ColorMode.DARK;

		// lazy load the mermaid dependecy only when needed to reduce bundle size.
		const { default: mermaid } = await import('mermaid');

		mermaid.initialize({
			startOnLoad: false,
			theme: isDark ? 'dark' : 'default',
			securityLevel: 'strict',
			flowchart: {
				useMaxWidth: false,
				htmlLabels: true
			},
			sequence: {
				useMaxWidth: false
			},
			gantt: {
				useMaxWidth: false
			}
		});

		try {
			await mermaid.run({
				nodes: Array.from(nodes) as unknown as NodeListOf<HTMLElement>
			});
		} catch (error) {
			console.error('Failed to render mermaid diagram:', error);
		}
	}

	/**
	 * Handles image load errors by replacing the image with a fallback UI.
	 * Shows a placeholder with a link to open the image in a new tab.
	 */
	function handleImageError(event: Event) {
		const img = event.target as HTMLImageElement;
		if (!img || !img.src) return;

		// Don't handle data URLs or already-handled images
		if (
			img.src.startsWith(UrlProtocol.DATA) ||
			img.dataset[DATA_ERROR_HANDLED_ATTR] === BOOL_TRUE_STRING
		)
			return;
		img.dataset[DATA_ERROR_HANDLED_ATTR] = BOOL_TRUE_STRING;

		const src = img.src;
		// Create fallback element
		const fallback = document.createElement('div');
		fallback.className = 'image-load-error';
		fallback.innerHTML = getImageErrorFallbackHtml(src);

		// Replace image with fallback
		img.parentNode?.replaceChild(fallback, img);
	}

	/**
	 * Queues markdown for processing with coalescing support.
	 * Only processes the latest markdown when multiple updates arrive quickly.
	 * Uses requestAnimationFrame to yield to browser paint between batches.
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

				// Yield to browser for paint. During this, new chunks coalesce
				// into pendingMarkdown, so we always render the latest state.
				if (pendingMarkdown !== null) {
					await new Promise((resolve) => requestAnimationFrame(resolve));
				}
			}
		} catch (error) {
			console.error('Failed to process markdown:', error);
			renderedBlocks = [];
			unstableBlockHtml = markdown.replace(/\n/g, '<br>');
		} finally {
			isProcessing = false;
		}
	}

	$effect(() => {
		const currentMode = mode.current;
		const isDark = currentMode === ColorMode.DARK;

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
			setupImageErrorHandlers();
			renderMermaidDiagrams();
		}
	});

	// Auto-scroll for streaming code block
	$effect(() => {
		streamingAutoScroll.setContainer(streamingCodeScrollContainer);
	});

	$effect(() => {
		streamingAutoScroll.updateInterval(incompleteCodeBlock !== null);
	});

	onDestroy(() => {
		cleanupEventListeners();
		cleanupHighlightTheme();
		streamingAutoScroll.destroy();
	});
</script>

<!-- svelte-ignore a11y_click_events_have_key_events -->
<!-- svelte-ignore a11y_no_static_element_interactions -->
<div
	bind:this={containerRef}
	onclick={handleMermaidClick}
	class="markdown-content {className}{config()[SETTINGS_KEYS.FULL_HEIGHT_CODE_BLOCKS]
		? ' full-height-code-blocks'
		: ''}"
>
	{#each renderedBlocks as block (block.id)}
		<div class="markdown-block" data-block-id={block.id} use:fadeInView={{ skipIfVisible: true }}>
			{@html block.html}
		</div>
	{/each}

	{#if unstableBlockHtml}
		<div class="markdown-block markdown-block--unstable" data-block-id="unstable">
			<!-- eslint-disable-next-line no-at-html-tags -->
			{@html unstableBlockHtml}
		</div>
	{/if}

	{#if incompleteCodeBlock}
		{#if incompleteCodeBlock.language === 'mermaid'}
			<div class="mermaid-block-wrapper streaming-mermaid-block">
				<div class="code-block-header">
					<span class="code-language">mermaid</span>
					<div class="code-block-actions">
						<ActionIconCopyToClipboard
							text={incompleteCodeBlock.code}
							canCopy={false}
							ariaLabel="Diagram incomplete"
						/>
					</div>
				</div>
				<div class="mermaid-loading-placeholder">
					<span class="mermaid-loading-text">Generating diagram...</span>
				</div>
			</div>
		{:else}
			<div class="code-block-wrapper streaming-code-block relative">
				<div class="code-block-header">
					<span class="code-language">{incompleteCodeBlock.language || 'text'}</span>
					<CodeBlockActions
						code={incompleteCodeBlock.code}
						language={incompleteCodeBlock.language || 'text'}
						disabled
						onPreview={(code, lang) => {
							previewCode = code;
							previewLanguage = lang;
							previewDialogOpen = true;
						}}
					/>
				</div>

				<div
					bind:this={streamingCodeScrollContainer}
					class="streaming-code-scroll-container"
					onscroll={() => streamingAutoScroll.handleScroll()}
				>
					<pre class="streaming-code-pre"><code
							class="hljs language-{incompleteCodeBlock.language || 'text'}"
							>{@html highlightCode(
								incompleteCodeBlock.code,
								incompleteCodeBlock.language || 'text'
							)}</code
						></pre>
				</div>
			</div>
		{/if}
	{/if}
</div>

<DialogCodePreview
	open={previewDialogOpen}
	code={previewCode}
	language={previewLanguage}
	onOpenChange={handlePreviewDialogOpenChange}
/>

<DialogMermaidPreview
	open={mermaidPreviewOpen}
	svgHtml={mermaidPreviewSvgHtml}
	onOpenChange={handleMermaidPreviewOpenChange}
/>

<style>
	@import './markdown-content.css';
</style>
