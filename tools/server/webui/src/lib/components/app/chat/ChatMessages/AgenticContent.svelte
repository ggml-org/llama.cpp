<script lang="ts">
	import { MarkdownContent } from '$lib/components/app';
	import { Wrench, ChevronDown, ChevronRight } from '@lucide/svelte';
	import { slide } from 'svelte/transition';

	interface Props {
		content: string;
	}

	interface AgenticSection {
		type: 'text' | 'tool_call';
		content: string;
		toolName?: string;
		toolArgs?: string;
		toolResult?: string;
	}

	let { content }: Props = $props();

	// Parse content into sections
	const sections = $derived(parseAgenticContent(content));

	// Track collapsed state for each tool call
	let collapsedArgs: Record<number, boolean> = $state({});

	function parseAgenticContent(rawContent: string): AgenticSection[] {
		if (!rawContent) return [];

		const sections: AgenticSection[] = [];
		const toolCallRegex =
			/<!-- AGENTIC_TOOL_CALL_START -->\n<!-- TOOL_NAME: (.+?) -->\n<!-- TOOL_ARGS: (.+?) -->\n([\s\S]*?)<!-- AGENTIC_TOOL_CALL_END -->/g;

		let lastIndex = 0;
		let match;

		while ((match = toolCallRegex.exec(rawContent)) !== null) {
			// Add text before this tool call
			if (match.index > lastIndex) {
				const textBefore = rawContent.slice(lastIndex, match.index).trim();
				if (textBefore) {
					sections.push({ type: 'text', content: textBefore });
				}
			}

			// Add tool call section
			const toolName = match[1];
			const toolArgs = match[2].replace(/\\n/g, '\n');
			const toolResult = match[3].trim();

			sections.push({
				type: 'tool_call',
				content: toolResult,
				toolName,
				toolArgs,
				toolResult
			});

			lastIndex = match.index + match[0].length;
		}

		// Add remaining text after last tool call
		if (lastIndex < rawContent.length) {
			const remainingText = rawContent.slice(lastIndex).trim();
			if (remainingText) {
				sections.push({ type: 'text', content: remainingText });
			}
		}

		// If no tool calls found, return content as single text section
		if (sections.length === 0 && rawContent.trim()) {
			sections.push({ type: 'text', content: rawContent });
		}

		return sections;
	}

	function toggleArgs(index: number) {
		collapsedArgs[index] = !collapsedArgs[index];
	}

	function formatToolArgs(args: string): string {
		try {
			const parsed = JSON.parse(args);
			return JSON.stringify(parsed, null, 2);
		} catch {
			return args;
		}
	}
</script>

<div class="agentic-content">
	{#each sections as section, index (index)}
		{#if section.type === 'text'}
			<div class="agentic-section agentic-text">
				<MarkdownContent content={section.content} />
			</div>
		{:else if section.type === 'tool_call'}
			<div class="agentic-section agentic-tool-call" transition:slide={{ duration: 200 }}>
				<div class="tool-call-header">
					<div class="tool-call-title">
						<Wrench class="h-4 w-4" />
						<span class="tool-name">{section.toolName}</span>
					</div>
					{#if section.toolArgs && section.toolArgs !== '{}'}
						<button
							type="button"
							class="tool-args-toggle"
							onclick={() => toggleArgs(index)}
							aria-expanded={!collapsedArgs[index]}
						>
							{#if collapsedArgs[index]}
								<ChevronRight class="h-4 w-4" />
							{:else}
								<ChevronDown class="h-4 w-4" />
							{/if}
							<span class="text-xs">Arguments</span>
						</button>
					{/if}
				</div>

				{#if section.toolArgs && section.toolArgs !== '{}' && !collapsedArgs[index]}
					<div class="tool-args" transition:slide={{ duration: 150 }}>
						<pre class="tool-args-content">{formatToolArgs(section.toolArgs)}</pre>
					</div>
				{/if}

				{#if section.toolResult}
					<div class="tool-result">
						<div class="tool-result-label">Result:</div>
						<MarkdownContent content={section.toolResult} />
					</div>
				{/if}
			</div>
		{/if}
	{/each}
</div>

<style>
	.agentic-content {
		display: flex;
		flex-direction: column;
		gap: 1rem;
	}

	.agentic-section {
		width: 100%;
	}

	.agentic-tool-call {
		border-left: 3px solid hsl(var(--primary) / 0.5);
		padding-left: 1rem;
		margin: 0.5rem 0;
	}

	.tool-call-header {
		display: flex;
		align-items: center;
		justify-content: space-between;
		gap: 0.5rem;
		margin-bottom: 0.5rem;
	}

	.tool-call-title {
		display: flex;
		align-items: center;
		gap: 0.5rem;
		font-weight: 600;
		color: hsl(var(--primary));
	}

	.tool-name {
		font-family: ui-monospace, SFMono-Regular, 'SF Mono', Monaco, monospace;
		font-size: 0.875rem;
	}

	.tool-args-toggle {
		display: flex;
		align-items: center;
		gap: 0.25rem;
		padding: 0.25rem 0.5rem;
		border-radius: 0.25rem;
		background: hsl(var(--muted) / 0.5);
		color: hsl(var(--muted-foreground));
		border: none;
		cursor: pointer;
		transition: background-color 0.15s;
	}

	.tool-args-toggle:hover {
		background: hsl(var(--muted));
	}

	.tool-args {
		margin: 0.5rem 0;
		padding: 0.5rem;
		background: hsl(var(--muted) / 0.3);
		border-radius: 0.375rem;
		overflow-x: auto;
	}

	.tool-args-content {
		margin: 0;
		font-family: ui-monospace, SFMono-Regular, 'SF Mono', Monaco, monospace;
		font-size: 0.75rem;
		color: hsl(var(--muted-foreground));
		white-space: pre-wrap;
		word-break: break-word;
	}

	.tool-result {
		margin-top: 0.5rem;
	}

	.tool-result-label {
		font-size: 0.75rem;
		font-weight: 500;
		color: hsl(var(--muted-foreground));
		margin-bottom: 0.25rem;
	}
</style>
