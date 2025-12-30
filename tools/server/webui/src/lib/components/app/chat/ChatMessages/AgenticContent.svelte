<script lang="ts">
	/**
	 * AgenticContent - Chronological display of agentic flow output
	 *
	 * Parses content with tool call markers and displays them inline
	 * with text content. Each tool call is shown as a collapsible box
	 * similar to the reasoning/thinking block UI.
	 */

	import { MarkdownContent } from '$lib/components/app';
	import { Wrench } from '@lucide/svelte';
	import ChevronsUpDownIcon from '@lucide/svelte/icons/chevrons-up-down';
	import * as Collapsible from '$lib/components/ui/collapsible/index.js';
	import { buttonVariants } from '$lib/components/ui/button/index.js';
	import { Card } from '$lib/components/ui/card';

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

	// Parse content into chronological sections
	const sections = $derived(parseAgenticContent(content));

	// Track expanded state for each tool call (default expanded)
	let expandedStates: Record<number, boolean> = $state({});

	function isExpanded(index: number): boolean {
		return expandedStates[index] ?? true;
	}

	function toggleExpanded(index: number) {
		expandedStates[index] = !isExpanded(index);
	}

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
			<div class="agentic-text">
				<MarkdownContent content={section.content} />
			</div>
		{:else if section.type === 'tool_call'}
			<Collapsible.Root open={isExpanded(index)} class="mb-4">
				<Card class="gap-0 border-muted bg-muted/30 py-0">
					<Collapsible.Trigger
						class="flex w-full cursor-pointer items-center justify-between p-3"
						onclick={() => toggleExpanded(index)}
					>
						<div class="flex items-center gap-2 text-muted-foreground">
							<Wrench class="h-4 w-4" />
							<span class="font-mono text-sm font-medium">{section.toolName}</span>
						</div>

						<div
							class={buttonVariants({
								variant: 'ghost',
								size: 'sm',
								class: 'h-6 w-6 p-0 text-muted-foreground hover:text-foreground'
							})}
						>
							<ChevronsUpDownIcon class="h-4 w-4" />
							<span class="sr-only">Toggle tool call content</span>
						</div>
					</Collapsible.Trigger>

					<Collapsible.Content>
						<div class="border-t border-muted px-3 pb-3">
							{#if section.toolArgs && section.toolArgs !== '{}'}
								<div class="pt-3">
									<div class="mb-1 text-xs text-muted-foreground">Arguments:</div>
									<pre
										class="rounded bg-muted/30 p-2 font-mono text-xs leading-relaxed break-words whitespace-pre-wrap">{formatToolArgs(
											section.toolArgs
										)}</pre>
								</div>
							{/if}

							{#if section.toolResult}
								<div class="pt-3">
									<div class="mb-1 text-xs text-muted-foreground">Result:</div>
									<div class="text-sm">
										<MarkdownContent content={section.toolResult} />
									</div>
								</div>
							{/if}
						</div>
					</Collapsible.Content>
				</Card>
			</Collapsible.Root>
		{/if}
	{/each}
</div>

<style>
	.agentic-content {
		display: flex;
		flex-direction: column;
		gap: 0.5rem;
		width: 100%;
		max-width: 48rem;
	}

	.agentic-text {
		width: 100%;
	}
</style>
