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

				try {
					// Decode HTML entities
					const decodedCode = rawCode
						.replace(/&amp;/g, '&')
						.replace(/&lt;/g, '<')
						.replace(/&gt;/g, '>')
						.replace(/&quot;/g, '"')
						.replace(/&#39;/g, "'");

					await navigator.clipboard.writeText(decodedCode);

					// Visual feedback
					target.innerHTML = `
						<svg class="w-4 h-4" fill="none" stroke="currentColor" viewBox="0 0 24 24">
							<path stroke-linecap="round" stroke-linejoin="round" stroke-width="2" d="M5 13l4 4L19 7"></path>
						</svg>
					`;

					// Reset icon after 2 seconds
					setTimeout(() => {
						target.innerHTML = `
							<svg class="w-4 h-4" fill="none" stroke="currentColor" viewBox="0 0 24 24">
								<path stroke-linecap="round" stroke-linejoin="round" stroke-width="2" d="M8 16H6a2 2 0 01-2-2V6a2 2 0 012-2h8a2 2 0 012 2v2m-6 12h8a2 2 0 002-2v-8a2 2 0 00-2-2h-8a2 2 0 00-2 2v8a2 2 0 002 2z"></path>
							</svg>
						`;
					}, 2000);
				} catch (error) {
					console.error('Failed to copy code:', error);
				}
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
			// Shiki code blocks (override default prose styles)
			'prose-pre:!bg-transparent prose-pre:!p-0 prose-pre:!m-0 prose-pre:!border-0',
			// Inline code
			'prose-code:bg-muted/70 prose-code:text-muted-foreground prose-code:px-1.5 prose-code:py-0.5 prose-code:rounded prose-code:text-sm prose-code:font-mono prose-code:before:content-none prose-code:after:content-none',
			// Links
			'prose-a:text-primary prose-a:hover:text-primary/80 prose-a:underline prose-a:underline-offset-2',
			// Lists
			'prose-ul:list-disc prose-ol:list-decimal prose-li:ml-4',
			// Blockquotes
			'prose-blockquote:border-l-4 prose-blockquote:border-muted prose-blockquote:pl-4 prose-blockquote:italic prose-blockquote:text-muted-foreground',
			// Headers
			'prose-h1:text-2xl prose-h1:font-bold prose-h1:mt-8 prose-h1:mb-4',
			'prose-h2:text-xl prose-h2:font-semibold prose-h2:mt-8 prose-h2:mb-4',
			'prose-h3:text-lg prose-h3:font-semibold prose-h3:mt-6 prose-h3:mb-3',
			// Paragraphs
			'prose-p:mb-4 prose-p:leading-relaxed',
			// Shiki syntax highlighting styles
			'[&_.shiki]:rounded-lg [&_.shiki]:rounded-lg [&_.shiki]:p-4 [&_.shiki]:pt-8 [&_.shiki]:overflow-x-auto',
			'[&_.shiki_code]:block [&_.shiki_code]:p-4 [&_.shiki_code]:block [&_.shiki_code]:text-sm [&_.shiki_code]:font-mono [&_.shiki_code]:leading-relaxed'
		].join(' ');

		return `${baseClasses} ${variantClasses} ${chatClasses} ${className}`;
	});
</script>

<div bind:this={containerRef} class={containerClasses}>
	{@html processedHtml}
</div>

<style>
	div :global(.pre) {
		border: none;
		padding: 1rem;
		border-radius: 2rem;
	}

	div :global(pre) {
		margin-block: 1.5rem;
		overflow-x: auto;
	}

	/* Code block wrapper styles */
	div :global(.code-block-wrapper) {
		position: relative;
		margin: 1.5rem 0;
		border-radius: 0.5rem;
		overflow: hidden;
		border: 1px solid hsl(var(--border));
	}

	/* Code block header styles */
	div :global(.code-block-header) {
		background: hsl(var(--muted));
		border-bottom: 1px solid hsl(var(--border));
		height: 3rem;
	}

	/* Language label styles */
	div :global(.code-block-header span) {
		font-size: 0.625rem;
		line-height: 0.75rem;
	}

	/* Copy button styles */
	div :global(.copy-code-btn) {
		border: none;
		background: none;
		cursor: pointer;
		display: flex;
		align-items: center;
		justify-content: center;
		border-radius: 0.25rem;
	}

	div :global(.copy-code-btn:hover) {
		background: hsl(var(--accent));
	}

	/* Code content styles */
	div :global(.code-content) {
		position: relative;
	}

	div :global(.code-content pre) {
		margin: 0;
		border-radius: 0;
		border: none;
	}

	div :global(.code-content .shiki) {
		border-radius: 0;
		margin: 0;
	}
</style>
