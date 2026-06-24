<script lang="ts">
	import { SyntaxHighlightedCode } from '$lib/components/app/content';
	import { computeLineDiff, formatUnifiedDiff, hasContentDiff } from '$lib/utils/diff';

	interface Props {
		/** Current/old content (e.g. the message text in the conversation). */
		oldContent: string;
		/** Incoming/new content (e.g. the latest prompt text from the library). */
		newContent: string;
		class?: string;
		maxHeight?: string;
	}

	let { oldContent, newContent, class: className = '', maxHeight = '50vh' }: Props = $props();

	let diffText = $derived.by(() => {
		if (!hasContentDiff(oldContent, newContent)) return '';
		return formatUnifiedDiff(computeLineDiff(oldContent, newContent));
	});
</script>

{#if diffText}
	<SyntaxHighlightedCode code={diffText} language="diff" {maxHeight} class={className} />
{:else}
	<div class="rounded-lg border border-border bg-muted p-4 text-sm text-muted-foreground">
		No differences.
	</div>
{/if}
