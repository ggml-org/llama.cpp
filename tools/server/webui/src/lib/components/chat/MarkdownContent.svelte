<script lang="ts">
	import { remark } from 'remark';
	import remarkHtml from 'remark-html';
	import remarkGfm from 'remark-gfm';
	import remarkBreaks from 'remark-breaks';
	import { onMount } from 'svelte';
	import { mode } from 'mode-watcher';
	import remarkShiki from '$lib/utils/remark-shiki';

	interface Props {
		content: string;
		class?: string;
		variant?: 'message' | 'thinking';
	}

	let { content, class: className = '', variant = 'message' }: Props = $props();

	let containerRef = $state<HTMLDivElement>();
	let processedHtml = $state('');

	// Configure remark processor with Shiki syntax highlighting
	const processor = $derived(() => {
		return remark()
			.use(remarkGfm) // GitHub Flavored Markdown
			.use(remarkBreaks) // Convert line breaks to <br>
			.use(remarkShiki, {
				theme: mode.current === 'dark' ? 'dark' : 'light'
			}) // Shiki syntax highlighting
			.use(remarkHtml, { sanitize: false }); // Convert to HTML
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

	// Initialize Shiki highlighter on mount
	onMount(async () => {
		try {
			// Initialize the highlighter to ensure it's ready
			const { initHighlighter } = await import('$lib/utils/syntax-highlighting');
			await initHighlighter();
		} catch (error) {
			console.error('Failed to initialize syntax highlighter:', error);
		}
	});

	// Dynamic classes based on variant
	const containerClasses = $derived(() => {
		const baseClasses = 'prose prose-sm dark:prose-invert max-w-none';
		const variantClasses =
			variant === 'thinking'
				? 'text-muted-foreground prose-headings:text-muted-foreground prose-strong:text-muted-foreground prose-code:text-muted-foreground prose-a:text-muted-foreground'
				: 'text-foreground';

		// Enhanced prose styling for better chat appearance
		const chatClasses = [
			// Base typography
			'prose-p:mb-4 prose-p:leading-relaxed prose-p:text-foreground',
			
			// Headers with consistent spacing
			'prose-h1:text-2xl prose-h1:font-bold prose-h1:mt-6 prose-h1:mb-3',
			'prose-h2:text-xl prose-h2:font-semibold prose-h2:mt-5 prose-h2:mb-2',
			'prose-h3:text-lg prose-h3:font-semibold prose-h3:mt-4 prose-h3:mb-2',
			'prose-h4:text-base prose-h4:font-semibold prose-h4:mt-3 prose-h4:mb-1',
			'prose-h5:text-base prose-h5:font-medium prose-h5:mt-2 prose-h5:mb-1',
			'prose-h6:text-sm prose-h6:font-medium prose-h6:mt-2 prose-h6:mb-1',
			
			// Text formatting
			'prose-strong:font-semibold prose-strong:text-foreground',
			'prose-em:italic prose-em:text-foreground',
			'prose-del:text-muted-foreground prose-del:no-underline',
			
			// Inline code
			'prose-code:bg-muted/50 prose-code:text-foreground prose-code:px-1.5 prose-code:py-0.5 prose-code:rounded-md prose-code:text-sm prose-code:font-mono prose-code:before:content-none prose-code:after:content-none',
			
			// Links
			'prose-a:text-primary prose-a:hover:text-primary/80 prose-a:underline prose-a:underline-offset-2 prose-a:transition-colors',
			
			// Lists
			'prose-ul:list-disc prose-ul:ml-4 prose-ul:mb-4',
			'prose-ol:list-decimal prose-ol:ml-4 prose-ol:mb-4',
			'prose-li:ml-2 prose-li:mb-1 prose-li:marker:text-muted-foreground',
			'prose-li:pl-2',
			
			// Nested lists
			'prose-ul:prose-ul:list-circle prose-ol:prose-ol:list-lower-alpha',
			
			// Task lists
			'[&_.task-list-item]:list-none [&_.task-list-item]:ml-0 [&_.task-list-item]:pl-0',
			'[&_.task-list-item-checkbox]:mr-2 [&_.task-list-item-checkbox]:mt-0.5',
			
			// Blockquotes
			'prose-blockquote:border-l-4 prose-blockquote:border-muted prose-blockquote:pl-4 prose-blockquote:py-1 prose-blockquote:italic prose-blockquote:text-muted-foreground prose-blockquote:bg-muted/20',
			
			// Tables
			'prose-table:w-full prose-table:my-4 prose-table:border-collapse',
			'prose-th:border prose-th:border-border prose-th:bg-muted/30 prose-th:px-3 prose-th:py-2 prose-th:text-left prose-th:font-semibold',
			'prose-td:border prose-td:border-border prose-td:px-3 prose-td:py-2',
			'prose-tr:nth-child(even):bg-muted/10',
			
			// Horizontal rules
			'prose-hr:border-border prose-hr:my-6',
			
			// Code blocks
			'prose-pre:!bg-transparent prose-pre:!p-0 prose-pre:!m-0 prose-pre:!border-0',
			
			// Shiki syntax highlighting
			'[&_.shiki]:rounded-lg [&_.shiki]:p-4 [&_.shiki]:pt-8 [&_.shiki]:overflow-x-auto [&_.shiki]:border [&_.shiki]:border-border',
			'[&_.shiki_code]:block [&_.shiki_code]:p-4 [&_.shiki_code]:text-sm [&_.shiki_code]:font-mono [&_.shiki_code]:leading-relaxed',
			
			// Mentions and hashtags
			'[&_.mention]:text-primary [&_.mention]:font-medium [&_.mention]:hover:underline',
			'[&_.hashtag]:text-primary [&_.hashtag]:font-medium [&_.hashtag]:hover:underline',
			
			// Images
			'prose-img:rounded-lg prose-img:shadow-sm prose-img:my-4',
			
			// Responsive design
			'max-w-none',
			'sm:prose-p:text-base',
			'sm:prose-h1:text-3xl',
			'sm:prose-h2:text-2xl',
			'sm:prose-h3:text-xl'
		].join(' ');

		return `${baseClasses} ${variantClasses} ${chatClasses} ${className}`;
	});
</script>

<div bind:this={containerRef} class={containerClasses}>
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
		margin: 1rem 0 0.5rem 0;
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
		background: hsl(var(--muted) / 0.5);
		color: hsl(var(--muted-foreground));
		padding: 0.125rem 0.375rem;
		border-radius: 0.375rem;
		font-size: 0.875rem;
		font-family: ui-monospace, SFMono-Regular, 'SF Mono', Monaco, 'Cascadia Code', 'Roboto Mono', Consolas, 'Liberation Mono', Menlo, monospace;
	}

	/* Links */
	div :global(a) {
		color: hsl(var(--primary));
		text-decoration: underline;
		text-underline-offset: 2px;
		transition: color 0.2s ease;
	}

	div :global(a:hover) {
		color: hsl(var(--primary) / 0.8);
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
		color: hsl(var(--muted-foreground));
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
		border-left: 4px solid hsl(var(--border));
		padding: 0.5rem 1rem;
		margin: 1rem 0;
		font-style: italic;
		color: hsl(var(--muted-foreground));
		background: hsl(var(--muted) / 0.2);
		border-radius: 0 0.375rem 0.375rem 0;
	}

	/* Tables */
	div :global(table) {
		width: 100%;
		margin: 1rem 0;
		border-collapse: collapse;
		border: 1px solid hsl(var(--border));
		border-radius: 0.375rem;
		overflow: hidden;
	}

	div :global(th) {
		background: hsl(var(--muted) / 0.3);
		border: 1px solid hsl(var(--border));
		padding: 0.5rem 0.75rem;
		text-align: left;
		font-weight: 600;
	}

	div :global(td) {
		border: 1px solid hsl(var(--border));
		padding: 0.5rem 0.75rem;
	}

	div :global(tr:nth-child(even)) {
		background: hsl(var(--muted) / 0.1);
	}

	/* Horizontal rules */
	div :global(hr) {
		border: none;
		border-top: 1px solid hsl(var(--border));
		margin: 1.5rem 0;
	}

	/* Images */
	div :global(img) {
		border-radius: 0.5rem;
		box-shadow: 0 1px 3px 0 rgb(0 0 0 / 0.1), 0 1px 2px -1px rgb(0 0 0 / 0.1);
		margin: 1rem 0;
		max-width: 100%;
		height: auto;
	}

	/* Code blocks */
	div :global(pre) {
		margin: 1.5rem 0;
		overflow-x: auto;
		border-radius: 0.5rem;
		background: transparent;
		padding: 0;
		border: none;
	}

	div :global(.pre) {
		border: none;
		padding: 1rem;
		border-radius: 0.5rem;
	}

	/* Shiki code blocks */
	div :global(.shiki) {
		border-radius: 0.5rem;
		border: 1px solid hsl(var(--border));
		overflow-x: auto;
		margin: 1.5rem 0;
		padding: 1rem;
		padding-top: 2rem;
		position: relative;
	}

	/* Code block wrapper */
	div :global(.code-block-wrapper) {
		position: relative;
		margin: 1.5rem 0;
		border-radius: 0.5rem;
		overflow: hidden;
		border: 1px solid hsl(var(--border));
		background: hsl(var(--background));
	}

	div :global(.code-block-header) {
		background: hsl(var(--muted));
		border-bottom: 1px solid hsl(var(--border));
		height: 2.5rem;
		display: flex;
		align-items: center;
		justify-content: space-between;
		padding: 0 1rem;
	}

	div :global(.code-block-header span) {
		font-size: 0.75rem;
		font-weight: 500;
		color: hsl(var(--muted-foreground));
		text-transform: uppercase;
		letter-spacing: 0.05em;
	}

	div :global(.copy-code-btn) {
		border: none;
		background: none;
		cursor: pointer;
		display: flex;
		align-items: center;
		justify-content: center;
		border-radius: 0.25rem;
		padding: 0.25rem;
		color: hsl(var(--muted-foreground));
		transition: all 0.2s ease;
	}

	div :global(.copy-code-btn:hover) {
		background: hsl(var(--accent));
		color: hsl(var(--accent-foreground));
	}

	div :global(.code-content) {
		position: relative;
	}

	div :global(.code-content pre) {
		margin: 0;
		border-radius: 0;
		border: none;
		background: transparent;
	}

	div :global(.code-content .shiki) {
		border-radius: 0;
		margin: 0;
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
	}
</style>
