<script lang="ts">
	import { remark } from 'remark';
	import remarkBreaks from 'remark-breaks';
	import remarkGfm from 'remark-gfm';
	import remarkMath from 'remark-math';
	import rehypeHighlight from 'rehype-highlight';
	import remarkRehype from 'remark-rehype';
	import rehypeKatex from 'rehype-katex';
	import rehypeStringify from 'rehype-stringify';
	import { copyCodeToClipboard } from '$lib/utils/copy';
	import { browser } from '$app/environment';
	import 'katex/dist/katex.min.css';

	import githubDarkCss from 'highlight.js/styles/github-dark.css?inline';
	import githubLightCss from 'highlight.js/styles/github.css?inline';
	import { mode } from 'mode-watcher';
	import { remarkLiteralHtml } from '$lib/markdown/literal-html';

	interface Props {
		content: string;
		class?: string;
	}

	let { content, class: className = '' }: Props = $props();

	let containerRef = $state<HTMLDivElement>();
	let processedHtml = $state('');

	function loadHighlightTheme(isDark: boolean) {
		if (!browser) return;

		const existingThemes = document.querySelectorAll('style[data-highlight-theme]');
		existingThemes.forEach((style) => style.remove());

		const style = document.createElement('style');
		style.setAttribute('data-highlight-theme', 'true');
		style.textContent = isDark ? githubDarkCss : githubLightCss;

		document.head.appendChild(style);
	}

	$effect(() => {
		const currentMode = mode.current;
		const isDark = currentMode === 'dark';

		loadHighlightTheme(isDark);
	});

	let processor = $derived(() => {
		return remark()
			.use(remarkGfm) // GitHub Flavored Markdown
			.use(remarkMath) // Parse $inline$ and $$block$$ math
			.use(remarkBreaks) // Convert line breaks to <br>
			.use(remarkLiteralHtml) // Treat raw HTML as literal text with preserved indentation
			.use(remarkRehype) // Convert Markdown AST to rehype
			.use(rehypeKatex) // Render math using KaTeX
			.use(rehypeHighlight) // Add syntax highlighting
			.use(rehypeStringify); // Convert to HTML string
	});

	function enhanceLinks(html: string): string {
		if (!html.includes('<a')) {
			return html;
		}

		const tempDiv = document.createElement('div');
		tempDiv.innerHTML = html;

		// Make all links open in new tabs
		const linkElements = tempDiv.querySelectorAll('a[href]');
		let mutated = false;

		for (const link of linkElements) {
			const target = link.getAttribute('target');
			const rel = link.getAttribute('rel');

			if (target !== '_blank' || rel !== 'noopener noreferrer') {
				mutated = true;
			}

			link.setAttribute('target', '_blank');
			link.setAttribute('rel', 'noopener noreferrer');
		}

		return mutated ? tempDiv.innerHTML : html;
	}

	function enhanceCodeBlocks(html: string): string {
		if (!html.includes('<pre')) {
			return html;
		}

		const tempDiv = document.createElement('div');
		tempDiv.innerHTML = html;

		const preElements = tempDiv.querySelectorAll('pre');
		let mutated = false;

		for (const [index, pre] of Array.from(preElements).entries()) {
			const codeElement = pre.querySelector('code');

			if (!codeElement) {
				continue;
			}

			mutated = true;

			let language = 'text';
			const classList = Array.from(codeElement.classList);

			for (const className of classList) {
				if (className.startsWith('language-')) {
					language = className.replace('language-', '');
					break;
				}
			}

			const rawCode = codeElement.textContent || '';
			const codeId = `code-${Date.now()}-${index}`;

			codeElement.setAttribute('data-code-id', codeId);
			codeElement.setAttribute('data-raw-code', rawCode);

			const wrapper = document.createElement('div');
			wrapper.className = 'code-block-wrapper';

			const header = document.createElement('div');
			header.className = 'code-block-header';

			const languageLabel = document.createElement('span');
			languageLabel.className = 'code-language';
			languageLabel.textContent = language;

			const copyButton = document.createElement('button');
			copyButton.className = 'copy-code-btn';
			copyButton.setAttribute('data-code-id', codeId);
			copyButton.setAttribute('title', 'Copy code');
			copyButton.setAttribute('type', 'button');

			copyButton.innerHTML = `
				<svg xmlns="http://www.w3.org/2000/svg" width="16" height="16" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2" stroke-linecap="round" stroke-linejoin="round" class="lucide lucide-copy-icon lucide-copy"><rect width="14" height="14" x="8" y="8" rx="2" ry="2"/><path d="M4 16c-1.1 0-2-.9-2-2V4c0-1.1.9-2 2-2h10c1.1 0 2 .9 2 2"/></svg>
			`;

			header.appendChild(languageLabel);
			header.appendChild(copyButton);
			wrapper.appendChild(header);

			const clonedPre = pre.cloneNode(true) as HTMLElement;
			wrapper.appendChild(clonedPre);

			pre.parentNode?.replaceChild(wrapper, pre);
		}

		return mutated ? tempDiv.innerHTML : html;
	}

	function normalizeMathDelimiters(text: string): string {
		return text
			.replace(/(^|[^\\])\\\[((?:\\.|[\s\S])*?)\\\]/g, (_, prefix: string, content: string) => {
				return `${prefix}$$${content}$$`;
			})
			.replace(/(^|[^\\])\\\(((?:\\.|[\s\S])*?)\\\)/g, (_, prefix: string, content: string) => {
				return `${prefix}$${content}$`;
			});
	}

	async function processMarkdown(text: string): Promise<string> {
		try {
			const normalized = normalizeMathDelimiters(text);
			const result = await processor().process(normalized);
			const html = String(result);
			const enhancedLinks = enhanceLinks(html);

			return enhanceCodeBlocks(enhancedLinks);
		} catch (error) {
			console.error('Markdown processing error:', error);

			// Fallback to plain text with line breaks
			return text.replace(/\n/g, '<br>');
		}
	}

	function setupCopyButtons() {
		if (!containerRef) return;

		const copyButtons = containerRef.querySelectorAll('.copy-code-btn');

		for (const button of copyButtons) {
			button.addEventListener('click', async (e) => {
				e.preventDefault();
				e.stopPropagation();

				const target = e.currentTarget as HTMLButtonElement;
				const codeId = target.getAttribute('data-code-id');

				if (!codeId) {
					console.error('No code ID found on button');
					return;
				}

				// Find the code element within the same wrapper
				const wrapper = target.closest('.code-block-wrapper');
				if (!wrapper) {
					console.error('No wrapper found');
					return;
				}

				const codeElement = wrapper.querySelector('code[data-code-id]');
				if (!codeElement) {
					console.error('No code element found in wrapper');
					return;
				}

				const rawCode = codeElement.getAttribute('data-raw-code');
				if (!rawCode) {
					console.error('No raw code found');
					return;
				}

				try {
					await copyCodeToClipboard(rawCode);
				} catch (error) {
					console.error('Failed to copy code:', error);
				}
			});
		}
	}

	$effect(() => {
		if (content) {
			processMarkdown(content)
				.then((result) => {
					processedHtml = result;
				})
				.catch((error) => {
					console.error('Failed to process markdown:', error);
					processedHtml = content.replace(/\n/g, '<br>');
				});
		} else {
			processedHtml = '';
		}
	});

	$effect(() => {
		if (containerRef && processedHtml) {
			setupCopyButtons();
		}
	});
</script>

<div bind:this={containerRef} class={className}>
	<!-- eslint-disable-next-line no-at-html-tags -->
	{@html processedHtml}
</div>

<style>
	/* Base typography styles */
	div :global(p:not(:last-child)) {
		margin-bottom: 1rem;
		line-height: 1.75;
	}

	/* Headers with consistent spacing */
	div :global(h1) {
		font-size: 1.875rem;
		font-weight: 700;
		margin: 1.5rem 0 0.75rem 0;
		line-height: 1.2;
	}

	div :global(h2) {
		font-size: 1.5rem;
		font-weight: 600;
		margin: 1.25rem 0 0.5rem 0;
		line-height: 1.3;
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

	div :global(.copy-code-btn) {
		display: flex;
		align-items: center;
		justify-content: center;
		padding: 0;
		background: transparent;
		color: var(--code-foreground);
		cursor: pointer;
		transition: all 0.2s ease;
	}

	div :global(.copy-code-btn:hover) {
		transform: scale(1.05);
	}

	div :global(.copy-code-btn:active) {
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
