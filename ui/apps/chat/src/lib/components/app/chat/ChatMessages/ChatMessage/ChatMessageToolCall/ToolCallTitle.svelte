<script lang="ts">
	// The title most tool rows share: what the call did, in muted text, then
	// what it acted on in monospace, then anything a tool adds after it.

	import type { Snippet } from 'svelte';

	interface Props {
		label: string;
		/** What the call acted on, e.g. a file name or a search term. */
		target?: string;
		/** Tooltip for the target, e.g. the full path behind a short one. */
		targetTitle?: string;
		/** Extras after the target, e.g. a line range or diff stats. */
		children?: Snippet;
	}

	let { children, label, target, targetTitle }: Props = $props();
</script>

<span class="flex min-w-0 flex-wrap items-baseline gap-x-1">
	<span class="shrink-0 text-muted-foreground">{label}</span>

	{#if target || children}
		<span class="flex min-w-0 items-baseline gap-1.5">
			{#if target}
				<span class="min-w-0 truncate font-mono" title={targetTitle ?? target}>{target}</span>
			{/if}

			{@render children?.()}
		</span>
	{/if}
</span>
