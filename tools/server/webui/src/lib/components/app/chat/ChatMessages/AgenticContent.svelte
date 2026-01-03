<script lang="ts">
	/**
	 * AgenticContent - Chronological display of agentic flow output
	 *
	 * Parses content with tool call markers and displays them inline
	 * with text content. Each tool call is shown as a collapsible box
	 * similar to the reasoning/thinking block UI.
	 */

	import { MarkdownContent, SyntaxHighlightedCode } from '$lib/components/app';
	import { config } from '$lib/stores/settings.svelte';
	import { Wrench, Loader2 } from '@lucide/svelte';
	import ChevronsUpDownIcon from '@lucide/svelte/icons/chevrons-up-down';
	import * as Collapsible from '$lib/components/ui/collapsible/index.js';
	import { buttonVariants } from '$lib/components/ui/button/index.js';
	import { Card } from '$lib/components/ui/card';

	interface Props {
		content: string;
	}

	interface AgenticSection {
		type: 'text' | 'tool_call' | 'tool_call_pending' | 'tool_call_streaming';
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

	// Get showToolCallInProgress setting
	const showToolCallInProgress = $derived(config().showToolCallInProgress as boolean);

	function isExpanded(index: number, isPending: boolean): boolean {
		// If showToolCallInProgress is enabled and tool is pending, force expand
		if (showToolCallInProgress && isPending) {
			return true;
		}
		// Otherwise use stored state, defaulting to expanded only if showToolCallInProgress is true
		return expandedStates[index] ?? showToolCallInProgress;
	}

	function toggleExpanded(index: number, isPending: boolean) {
		expandedStates[index] = !isExpanded(index, isPending);
	}

	function parseAgenticContent(rawContent: string): AgenticSection[] {
		if (!rawContent) return [];

		const sections: AgenticSection[] = [];

		// Regex for completed tool calls (with END marker)
		const completedToolCallRegex =
			/<<<AGENTIC_TOOL_CALL_START>>>\n<<<TOOL_NAME:(.+?)>>>\n<<<TOOL_ARGS_BASE64:(.+?)>>>([\s\S]*?)<<<AGENTIC_TOOL_CALL_END>>>/g;

		let lastIndex = 0;
		let match;

		// First pass: find all completed tool calls
		while ((match = completedToolCallRegex.exec(rawContent)) !== null) {
			// Add text before this tool call
			if (match.index > lastIndex) {
				const textBefore = rawContent.slice(lastIndex, match.index).trim();
				if (textBefore) {
					sections.push({ type: 'text', content: textBefore });
				}
			}

			// Add completed tool call section
			const toolName = match[1];
			const toolArgsBase64 = match[2];
			let toolArgs = '';
			try {
				toolArgs = decodeURIComponent(escape(atob(toolArgsBase64)));
			} catch {
				toolArgs = toolArgsBase64;
			}
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

		// Check for pending tool call at the end (START without END)
		const remainingContent = rawContent.slice(lastIndex);

		// Full pending match (has NAME and ARGS)
		const pendingMatch = remainingContent.match(
			/<<<AGENTIC_TOOL_CALL_START>>>\n<<<TOOL_NAME:(.+?)>>>\n<<<TOOL_ARGS_BASE64:(.+?)>>>([\s\S]*)$/
		);

		// Partial pending match (has START and NAME but ARGS still streaming)
		const partialWithNameMatch = remainingContent.match(
			/<<<AGENTIC_TOOL_CALL_START>>>\n<<<TOOL_NAME:(.+?)>>>\n<<<TOOL_ARGS_BASE64:([^>]*)$/
		);

		// Very early match (just START marker, maybe partial NAME)
		const earlyMatch = remainingContent.match(/<<<AGENTIC_TOOL_CALL_START>>>([\s\S]*)$/);

		if (pendingMatch) {
			// Add text before pending tool call
			const pendingIndex = remainingContent.indexOf('<<<AGENTIC_TOOL_CALL_START>>>');
			if (pendingIndex > 0) {
				const textBefore = remainingContent.slice(0, pendingIndex).trim();
				if (textBefore) {
					sections.push({ type: 'text', content: textBefore });
				}
			}

			// Add pending tool call
			const toolName = pendingMatch[1];
			const toolArgsBase64 = pendingMatch[2];
			let toolArgs = '';
			try {
				toolArgs = decodeURIComponent(escape(atob(toolArgsBase64)));
			} catch {
				toolArgs = toolArgsBase64;
			}
			// Capture streaming result content (everything after args marker)
			const streamingResult = pendingMatch[3]?.trim() || '';

			sections.push({
				type: 'tool_call_pending',
				content: streamingResult,
				toolName,
				toolArgs,
				toolResult: streamingResult || undefined
			});
		} else if (partialWithNameMatch) {
			// Has START and NAME, ARGS still streaming
			const pendingIndex = remainingContent.indexOf('<<<AGENTIC_TOOL_CALL_START>>>');
			if (pendingIndex > 0) {
				const textBefore = remainingContent.slice(0, pendingIndex).trim();
				if (textBefore) {
					sections.push({ type: 'text', content: textBefore });
				}
			}

			sections.push({
				type: 'tool_call_streaming',
				content: '',
				toolName: partialWithNameMatch[1],
				toolArgs: undefined,
				toolResult: undefined
			});
		} else if (earlyMatch) {
			// Just START marker, show streaming state
			const pendingIndex = remainingContent.indexOf('<<<AGENTIC_TOOL_CALL_START>>>');
			if (pendingIndex > 0) {
				const textBefore = remainingContent.slice(0, pendingIndex).trim();
				if (textBefore) {
					sections.push({ type: 'text', content: textBefore });
				}
			}

			// Try to extract tool name if present
			const nameMatch = earlyMatch[1]?.match(/<<<TOOL_NAME:([^>]+)>>>/);

			sections.push({
				type: 'tool_call_streaming',
				content: '',
				toolName: nameMatch?.[1],
				toolArgs: undefined,
				toolResult: undefined
			});
		} else if (lastIndex < rawContent.length) {
			// Add remaining text after last completed tool call
			// But strip any partial markers that might be starting
			let remainingText = rawContent.slice(lastIndex).trim();

			// Check for partial marker at the end (e.g., "<<<" or "<<<AGENTIC" etc.)
			const partialMarkerMatch = remainingText.match(/<<<[A-Z_]*$/);
			if (partialMarkerMatch) {
				remainingText = remainingText.slice(0, partialMarkerMatch.index).trim();
			}

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

	function isJsonContent(content: string): boolean {
		const trimmed = content.trim();
		return (
			(trimmed.startsWith('{') && trimmed.endsWith('}')) ||
			(trimmed.startsWith('[') && trimmed.endsWith(']'))
		);
	}

	function formatJsonContent(content: string): string {
		try {
			const parsed = JSON.parse(content);
			return JSON.stringify(parsed, null, 2);
		} catch {
			return content;
		}
	}
</script>

<div class="agentic-content">
	{#each sections as section, index (index)}
		{#if section.type === 'text'}
			<div class="agentic-text">
				<MarkdownContent content={section.content} />
			</div>
		{:else if section.type === 'tool_call_streaming'}
			<!-- Early streaming state - show minimal UI while markers are being received -->
			<div class="my-4">
				<Card class="gap-0 border-muted bg-muted/30 py-0">
					<div class="flex items-center gap-2 p-3 text-muted-foreground">
						<Loader2 class="h-4 w-4 animate-spin" />
						{#if section.toolName}
							<span class="font-mono text-sm font-medium">{section.toolName}</span>
						{/if}
						<span class="text-xs italic">preparing...</span>
					</div>
				</Card>
			</div>
		{:else if section.type === 'tool_call' || section.type === 'tool_call_pending'}
			{@const isPending = section.type === 'tool_call_pending'}
			<Collapsible.Root open={isExpanded(index, isPending)} class="my-2">
				<Card class="gap-0 border-muted bg-muted/30 py-0">
					<Collapsible.Trigger
						class="flex w-full cursor-pointer items-center justify-between p-3"
						onclick={() => toggleExpanded(index, isPending)}
					>
						<div class="flex items-center gap-2 text-muted-foreground">
							{#if isPending}
								<Loader2 class="h-4 w-4 animate-spin" />
							{:else}
								<Wrench class="h-4 w-4" />
							{/if}
							<span class="font-mono text-sm font-medium">{section.toolName}</span>
							{#if isPending}
								<span class="text-xs italic">executing...</span>
							{/if}
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
									<div class="my-3 text-xs text-muted-foreground">Arguments:</div>
									<SyntaxHighlightedCode
										code={formatToolArgs(section.toolArgs)}
										language="json"
										maxHeight="20rem"
										class="text-xs"
									/>
								</div>
							{/if}

							<div class="pt-3">
								<div class="my-3 flex items-center gap-2 text-xs text-muted-foreground">
									<span>Result:</span>
									{#if isPending}
										<Loader2 class="h-3 w-3 animate-spin" />
									{/if}
								</div>
								{#if section.toolResult}
									{#if isJsonContent(section.toolResult)}
										<SyntaxHighlightedCode
											code={formatJsonContent(section.toolResult)}
											language="json"
											maxHeight="20rem"
											class="text-xs"
										/>
									{:else}
										<div class="text-sm">
											<MarkdownContent content={section.toolResult} disableMath />
										</div>
									{/if}
								{:else if isPending}
									<div class="rounded bg-muted/30 p-2 text-xs text-muted-foreground italic">
										Waiting for result...
									</div>
								{/if}
							</div>
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
