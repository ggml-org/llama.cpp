<script lang="ts">
	import { remark } from 'remark';
	import remarkGfm from 'remark-gfm';
	import remarkBreaks from 'remark-breaks';
	import remarkRehype from 'remark-rehype';
	import rehypeHighlight from 'rehype-highlight';
	import rehypeStringify from 'rehype-stringify';
	// Import highlight.js CSS theme
	import 'highlight.js/styles/github-dark-dimmed.css';

	interface Props {
		content: string;
		class?: string;
	}

	let { content, class: className = '' }: Props = $props();

	let containerRef = $state<HTMLDivElement>();
	let processedHtml = $state('');

	// Configure remark processor with rehype-highlight syntax highlighting
	const processor = $derived(() => {
		return remark()
			.use(remarkGfm) // GitHub Flavored Markdown
			.use(remarkBreaks) // Convert line breaks to <br>
			.use(remarkRehype) // Convert to rehype (HTML AST)
			.use(rehypeHighlight) // Add syntax highlighting
			.use(rehypeStringify); // Convert to HTML string
	});

	// Process markdown content with syntax highlighting
	async function processMarkdown(text: string): Promise<string> {
		try {
			const result = await processor().process(text);
			return String(result);
		} catch (error) {
			console.error('Markdown processing error:', error);
			// Fallback to plain text with line breaks
			return text.replace(/\n/g, '<br>');
		}
	}

	// Function to setup copy-to-clipboard buttons
	function setupCopyButtons() {
		if (!containerRef) return;

		const copyButtons = containerRef.querySelectorAll('.copy-code-btn');
		copyButtons.forEach((button) => {
			button.addEventListener('click', async (e) => {
				const target = e.currentTarget as HTMLButtonElement;
				const codeId = target.getAttribute('data-code-id');
				if (!codeId) return;

				const codeContent = containerRef!.querySelector(`[data-code-id="${codeId}"]`);
				if (!codeContent) return;

				const rawCode = codeContent.getAttribute('data-raw-code');
				if (!rawCode) return;

				// Use the reusable copy function
				const { copyCodeToClipboard } = await import('$lib/utils/copy');
				await copyCodeToClipboard(rawCode);
			});
		});
	}

	// Update processed content when content or theme changes
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

	// Setup copy-to-clipboard functionality after content is rendered
	$effect(() => {
		if (containerRef && processedHtml) {
			setupCopyButtons();
		}
	});
</script>

<div bind:this={containerRef} class={className}>
	{@html processedHtml}
</div>

<style>
	/* Base typography styles */
	div :global(p) {
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
		font-family: ui-monospace, SFMono-Regular, 'SF Mono', Monaco, 'Cascadia Code', 'Roboto Mono', Consolas, 'Liberation Mono', Menlo, monospace;
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
		box-shadow: 0 1px 3px 0 rgb(0 0 0 / 0.1), 0 1px 2px -1px rgb(0 0 0 / 0.1);
		margin: 1.5rem 0;
		max-width: 100%;
		height: auto;
	}

	/* Code blocks */
	div :global(pre) {
		background: var(--muted);
		/* padding: 1.5rem 2rem; */
		margin: 1.5rem 0;
		overflow-x: auto;
		border-radius: 1rem;
		border: none;
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
		box-shadow: 0 4px 6px -1px rgb(0 0 0 / 0.1), 0 2px 4px -2px rgb(0 0 0 / 0.1);
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
		box-shadow: 0 10px 15px -3px rgb(0 0 0 / 0.1), 0 4px 6px -4px rgb(0 0 0 / 0.1);
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
