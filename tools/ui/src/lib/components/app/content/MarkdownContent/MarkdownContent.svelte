<script lang="ts">
	import '$lib/styles/katex-custom.scss';
	import {
		getDiagramButtonAction,
		renderMermaidDiagrams,
		renderSvgDiagrams,
		toggleDiagramSource
	} from './markdown-diagrams';
	import { getMarkdownProcessor, type MarkdownProcessor } from './markdown-processor';
	import {
		getCodeInfoFromTarget,
		getHastNodeId,
		getMdastNodeHash,
		isAppendMode
	} from './markdown-utils';
	import MarkdownSurface from './MarkdownSurface.svelte';
	import { SAFE_HTML_CONFIG } from './safe-html-config';
	import {
		ActionIconCopyToClipboard,
		CodeBlockActions,
		DialogCodePreview,
		DialogMermaidPreview
	} from '$lib/components/app';
	import {
		CODE_BLOCK_CLASS,
		IMAGE_NOT_ERROR_BOUND_SELECTOR,
		MARKDOWN_DATA_ATTRS,
		MERMAID_BLOCK_CLASS,
		MERMAID_LANGUAGE,
		SETTINGS_KEYS,
		SVG
	} from '$lib/constants';
	import { BooleanString, ColorMode, UrlProtocol } from '$lib/enums';
	import { createStickToBottom } from '$lib/hooks/use-stick-to-bottom.svelte';
	import { settingsStore } from '$lib/stores';
	import type { DatabaseMessageExtra } from '$lib/types/database';
	import {
		copyCodeToClipboard,
		copyToClipboard,
		getImageErrorFallbackHtml,
		loadHighlightTheme,
		preprocessLaTeX,
		splitGluedClosingCodeFences
	} from '$lib/utils';
	import { detectIncompleteCodeBlock, highlightCode, type IncompleteCodeBlock } from '$lib/utils';
	import { sanitizeSvg } from '$lib/utils/sanitize-svg';
	import { mountSvgShadow } from '$lib/utils/svg-shadow';
	import DOMPurify from 'dompurify';
	import type { Root as HastRoot, RootContent as HastRootContent } from 'hast';
	import type { Root as MdastRoot } from 'mdast';
	import { mode } from 'mode-watcher';
	import { onDestroy, tick } from 'svelte';
	import { SvelteMap } from 'svelte/reactivity';

	interface Props {
		attachments?: DatabaseMessageExtra[];
		content: string;
		class?: string;
		disableMath?: boolean;
		/** Render raw HTML found in the markdown (sanitized) instead of escaping it. */
		allowHtml?: boolean;
	}

	interface MarkdownBlock {
		id: string;
		html: string;
		contentHash?: string;
	}

	let {
		allowHtml = false,
		attachments,
		class: className = '',
		content,
		disableMath = false
	}: Props = $props();

	let containerRef = $state<HTMLDivElement>();
	let renderedBlocks = $state<MarkdownBlock[]>([]);
	let unstableBlockHtml = $state('');
	let incompleteCodeBlock = $state<IncompleteCodeBlock | null>(null);
	const streamingSvgCode = $derived.by(() => {
		const block = incompleteCodeBlock;

		if (!block) return null;

		if (block.language === SVG.LANGUAGE) return block.code;

		if (block.language === SVG.XML_LANGUAGE && block.code.trimStart().startsWith(SVG.TAG_PREFIX))
			return block.code;

		return null;
	});
	const liveSvgHtml = $derived(streamingSvgCode !== null ? sanitizeSvg(streamingSvgCode) : '');

	// Derived rather than called inline in the template so it only recomputes when
	// the block actually changes. Auto-detection is disabled while streaming: it
	// costs ~38ms a call and re-guesses the language on every chunk.
	const streamingCodeHtml = $derived(
		incompleteCodeBlock
			? highlightCode(incompleteCodeBlock.code, incompleteCodeBlock.language || 'text', false)
			: ''
	);
	let previewDialogOpen = $state(false);
	let previewCode = $state('');
	let previewLanguage = $state('text');
	let mermaidPreviewOpen = $state(false);
	let mermaidPreviewSvgHtml = $state('');
	let svgPreviewLive = $state(false);
	let streamingSvgHost = $state<HTMLDivElement | null>(null);

	// While the zoom dialog is open on a streaming svg, mirror the live render into it
	$effect(() => {
		if (svgPreviewLive && liveSvgHtml) mermaidPreviewSvgHtml = liveSvgHtml;
	});

	// Mount the streaming svg into its shadow host on every chunk so it renders live
	$effect(() => {
		if (streamingSvgHost) mountSvgShadow(streamingSvgHost, liveSvgHtml, SVG.INLINE_SHADOW_STYLE);
	});

	let streamingCodeScrollContainer = $state<HTMLDivElement>();

	// Follows the code that streams into the fixed-height block below.
	const streamingScroll = createStickToBottom();

	let pendingMarkdown: string | null = null;
	let isProcessing = false;

	// Per-instance transform cache, avoids re-transforming stable blocks during streaming
	// Garbage collected when component is destroyed (on conversation change)
	const CACHE_KEY_PLAIN = 'plain';
	const CACHE_KEY_ALLOW_HTML = 'allow-html';
	const transformCache = new SvelteMap<string, string>();
	let previousContent = '';

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
		processorInstance: MarkdownProcessor,
		node: unknown,
		index: number
	): Promise<{ html: string; hash: string }> {
		const hash = getMdastNodeHash(node, index);
		// the rendered HTML also depends on allowHtml (sanitized raw vs escaped),
		// so the cache is keyed per mode
		const cacheKey = `${allowHtml ? CACHE_KEY_ALLOW_HTML : CACHE_KEY_PLAIN}:${hash}`;
		const cached = transformCache.get(cacheKey);

		if (cached) {
			return { hash, html: cached };
		}

		const singleNodeRoot = { children: [node], type: 'root' };
		const transformedRoot = (await processorInstance.run(singleNodeRoot as MdastRoot)) as HastRoot;
		const html = processorInstance.stringify(transformedRoot);
		const safeHtml = allowHtml
			? (DOMPurify.sanitize(html, SAFE_HTML_CONFIG) as unknown as string)
			: html;

		transformCache.set(cacheKey, safeHtml);

		return { hash, html: safeHtml };
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
	async function processMarkdown(rawMarkdown: string) {
		// Text glued to a closing code fence is not a fence to the parser -
		// the block would swallow it. Split it onto its own line first.
		const markdown = splitGluedClosingCodeFences(rawMarkdown);

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
				const processorInstance = getMarkdownProcessor({ allowHtml, attachments, disableMath });
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
					const { hash, html } = await transformMdastNode(processorInstance, child, index);
					const id = getHastNodeId(
						{ position: (child as { position?: unknown }).position } as HastRootContent,
						index
					);

					nextBlocks.push({ contentHash: hash, html, id });
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
		const processorInstance = getMarkdownProcessor({ allowHtml, attachments, disableMath });
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
			const { hash, html } = await transformMdastNode(processorInstance, child, index);
			const id = getHastNodeId(
				{ position: (child as { position?: unknown }).position } as HastRootContent,
				index
			);

			nextBlocks.push({ contentHash: hash, html, id });
		}

		let unstableHtml = '';

		if (mdastChildren.length > stableCount) {
			const unstableChild = mdastChildren[stableCount];
			const singleNodeRoot = { children: [unstableChild], type: 'root' };
			const transformedRoot = (await processorInstance.run(
				singleNodeRoot as MdastRoot
			)) as HastRoot;

			unstableHtml = processorInstance.stringify(transformedRoot);

			if (allowHtml) {
				unstableHtml = DOMPurify.sanitize(unstableHtml, SAFE_HTML_CONFIG) as unknown as string;
			}
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

			if (
				copyButton &&
				copyButton.getAttribute(MARKDOWN_DATA_ATTRS.LISTENER_BOUND) !== BooleanString.TRUE
			) {
				copyButton.setAttribute(MARKDOWN_DATA_ATTRS.LISTENER_BOUND, BooleanString.TRUE);
				copyButton.addEventListener('click', handleCopyClick);
			}

			if (
				previewButton &&
				previewButton.getAttribute(MARKDOWN_DATA_ATTRS.LISTENER_BOUND) !== BooleanString.TRUE
			) {
				previewButton.setAttribute(MARKDOWN_DATA_ATTRS.LISTENER_BOUND, BooleanString.TRUE);
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
			img.setAttribute(MARKDOWN_DATA_ATTRS.ERROR_BOUND, BooleanString.TRUE);
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

		if (toggleDiagramSource(target)) {
			event.preventDefault();
			event.stopPropagation();

			return;
		}

		const action = getDiagramButtonAction(target);

		if (action) {
			event.preventDefault();
			event.stopPropagation();

			if ('copy' in action) {
				try {
					await copyToClipboard(action.copy);
				} catch (error) {
					console.error('Failed to copy diagram source:', error);
				}

				return;
			}

			mermaidPreviewSvgHtml = action.previewSvg;
			svgPreviewLive = false;
			mermaidPreviewOpen = true;

			return;
		}

		// A click on the header chrome targets the action buttons, never the
		// diagram. Guard so a header click can not fall through to the click to
		// zoom branches below, whatever the scroll position or stacking.
		if (target.closest(`.${CODE_BLOCK_CLASS.HEADER}`)) return;

		// Open preview when clicking the svg block itself. A final block carries its
		// source, a streaming block does not and is mirrored live into the dialog.
		const svgEl = target.closest(`.${SVG.BLOCK_CLASS}`);

		if (svgEl) {
			const source = svgEl.getAttribute(SVG.SOURCE_ATTR);

			if (source !== null) {
				mermaidPreviewSvgHtml = sanitizeSvg(source);
				svgPreviewLive = false;
			} else {
				svgPreviewLive = true;
			}

			mermaidPreviewOpen = true;

			return;
		}

		// Otherwise, open preview when clicking on the mermaid diagram itself
		const mermaidEl = target.closest(`.${MERMAID_BLOCK_CLASS}`);

		if (!mermaidEl) return;

		const svg = mermaidEl.querySelector('svg');

		if (!svg) return;

		mermaidPreviewSvgHtml = svg.outerHTML;
		svgPreviewLive = false;
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
			svgPreviewLive = false;
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
			img.getAttribute(MARKDOWN_DATA_ATTRS.ERROR_HANDLED) === BooleanString.TRUE
		)
			return;

		img.setAttribute(MARKDOWN_DATA_ATTRS.ERROR_HANDLED, BooleanString.TRUE);

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
			renderMermaidDiagrams(containerRef, mode.current === ColorMode.DARK);
			renderSvgDiagrams(containerRef);
		}
	});

	// Follow the streaming code block while the reader stays at its bottom; the
	// resize observer does the pinning, so no polling for a stable height.
	$effect(() => {
		streamingScroll.attach(streamingCodeScrollContainer);

		return () => streamingScroll.detach();
	});

	onDestroy(() => {
		cleanupEventListeners();
	});
</script>

<MarkdownSurface
	bind:ref={containerRef}
	class="{className}{settingsStore.config[SETTINGS_KEYS.FULL_HEIGHT_CODE_BLOCKS]
		? ' full-height-code-blocks'
		: ''}"
	onclick={handleMermaidClick}
>
	{#each renderedBlocks as block (block.id)}
		<div class="markdown-block" {...{ [MARKDOWN_DATA_ATTRS.BLOCK_ID]: block.id }}>
			{@html block.html}
		</div>
	{/each}

	{#if unstableBlockHtml}
		<div
			class="markdown-block markdown-block--unstable"
			{...{ [MARKDOWN_DATA_ATTRS.BLOCK_ID]: 'unstable' }}
		>
			<!-- eslint-disable-next-line no-at-html-tags -->
			{@html unstableBlockHtml}
		</div>
	{/if}

	{#if incompleteCodeBlock}
		{#if incompleteCodeBlock.language === MERMAID_LANGUAGE}
			<div class="mermaid-block-wrapper streaming-mermaid-block">
				<div class="code-block-header">
					<span class="code-language">mermaid</span>

					<div class="code-block-actions">
						<ActionIconCopyToClipboard
							ariaLabel="Diagram incomplete"
							canCopy={false}
							text={incompleteCodeBlock.code}
						/>
					</div>
				</div>

				<div class="mermaid-loading-placeholder">
					<span class="mermaid-loading-text">Generating diagram...</span>
				</div>
			</div>
		{:else if streamingSvgCode !== null}
			<div class="svg-block-wrapper streaming-svg-block">
				<div class="code-block-header">
					<span class="code-language">svg</span>

					<div class="code-block-actions">
						<ActionIconCopyToClipboard
							ariaLabel="Diagram incomplete"
							canCopy={false}
							text={incompleteCodeBlock.code}
						/>
					</div>
				</div>

				{#if liveSvgHtml}
					<div class="svg-scroll-container">
						<div class={SVG.BLOCK_CLASS}>
							<div bind:this={streamingSvgHost}></div>
						</div>
					</div>
				{:else}
					<div class="mermaid-loading-placeholder">
						<span class="mermaid-loading-text">Rendering svg...</span>
					</div>
				{/if}
			</div>
		{:else}
			<div class="code-block-wrapper streaming-code-block relative">
				<div class="code-block-header">
					<span class="code-language">{incompleteCodeBlock.language || 'text'}</span>

					<CodeBlockActions
						code={incompleteCodeBlock.code}
						disabled
						language={incompleteCodeBlock.language || 'text'}
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
					onscroll={() => streamingScroll.handleScroll()}
				>
					<pre class="streaming-code-pre"><code
							class="hljs language-{incompleteCodeBlock.language || 'text'}"
							>{@html streamingCodeHtml}</code
						></pre>
				</div>
			</div>
		{/if}
	{/if}
</MarkdownSurface>

<DialogCodePreview
	code={previewCode}
	language={previewLanguage}
	onOpenChange={handlePreviewDialogOpenChange}
	open={previewDialogOpen}
/>

<DialogMermaidPreview
	onOpenChange={handleMermaidPreviewOpenChange}
	open={mermaidPreviewOpen}
	svgHtml={mermaidPreviewSvgHtml}
/>
