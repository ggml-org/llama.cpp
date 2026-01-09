<script lang="ts">
	import type { DatabaseMessage } from '$lib/types';

	interface Props {
		class?: string;
		message: DatabaseMessage;
	}

	let { class: className = '', message }: Props = $props();

	const parseContent = () => {
		try {
			return JSON.parse(message.content);
		} catch {
			return null;
		}
	};

	const parsed = $derived(parseContent());
</script>

<div class={`rounded-md border bg-muted/40 p-3 text-sm text-foreground ${className}`}>
	<div class="mb-1 text-xs font-semibold text-muted-foreground uppercase">Tool • Calculator</div>
	{#if parsed && typeof parsed === 'object'}
		{#if parsed.expression}
			<div class="mb-1 text-xs tracking-wide text-muted-foreground uppercase">Expression</div>
			<div class="rounded-sm bg-background/70 px-2 py-1 font-mono text-xs">
				{parsed.expression}
			</div>
		{/if}
		{#if parsed.result !== undefined}
			<div class="mt-2 mb-1 text-xs tracking-wide text-muted-foreground uppercase">Result</div>
			<div class="rounded-sm bg-background/70 px-2 py-1 font-mono text-xs">
				{parsed.result}
			</div>
		{/if}
	{:else}
		<div
			class="rounded-sm bg-background/70 px-2 py-1 font-mono text-xs break-words whitespace-pre-wrap"
		>
			{message.content}
		</div>
	{/if}
</div>
