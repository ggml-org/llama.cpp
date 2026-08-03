<script lang="ts">
	import type { AgenticSection } from '$lib/utils';
	import { parseQuestionToolArguments, parseQuestionToolResult } from './parsers/question';
	import ToolCallBlock from './ToolCallBlock.svelte';

	interface Props {
		section: AgenticSection;
		open: boolean;
		isStreaming: boolean;
		onToggle?: () => void;
	}

	let { section, open, isStreaming, onToggle }: Props = $props();

	const questions = $derived(parseQuestionToolArguments(section.toolArgs));
	const answerPairs = $derived(parseQuestionToolResult(section.toolResult, questions));
	const wasDismissed = $derived(section.toolResult === 'The user dismissed this question.');
	const questionMeta = $derived({
		answerPairs,
		errorMessage: wasDismissed ? 'Question dismissed' : undefined
	});
</script>

<ToolCallBlock
	{section}
	{open}
	{isStreaming}
	meta={questionMeta}
	title={answerPairs.length > 0 ? 'Question answered' : 'Question'}
	{onToggle}
>
	{#snippet children(meta, _ctx)}
		{#if meta?.answerPairs.length}
			<div class="space-y-2">
				{#each meta.answerPairs as pair, index (`${index}:${pair.question}`)}
					<div class="bg-muted/30 rounded-md px-2 py-1.5">
						<div class="break-words text-sm leading-5">{pair.question}</div>
						<div class="text-muted-foreground mt-1 break-words text-sm leading-5">
							{pair.answer || 'No answer'}
						</div>
					</div>
				{/each}
			</div>
		{:else if wasDismissed}
			<div class="text-muted-foreground text-sm">Question dismissed.</div>
		{:else if section.toolResult}
			<div class="text-muted-foreground text-sm">Answer unavailable.</div>
		{:else}
			<div class="text-muted-foreground text-sm">Waiting for an answer...</div>
		{/if}
	{/snippet}
</ToolCallBlock>
