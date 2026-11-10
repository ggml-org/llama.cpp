<!--
	One bit depth of a model: its label ("4-bit"), the memory its quants need, and their chips. The props-in version of
	ui/apps/chat's ModelsDiscoverDetailsDownloadOptionsRow: the chips come in as children, and a `command` snippet, when
	given, opens under the row from its label, the way the app shows the command for that quant.
-->
<script lang="ts">
	import { ChevronDown } from '@lucide/svelte';
	import {
		BIT_DEPTH_LABEL_SUFFIX,
		GIGABYTE_LABEL,
		OTHER_BIT_DEPTH,
		OTHER_BIT_DEPTH_LABEL
	} from '$lib/models/constants';
	import type { Snippet } from 'svelte';
	import { cubicOut } from 'svelte/easing';
	import { slide } from 'svelte/transition';

	interface Props {
		/** Bit depth of the row; `99` renders as "Other". */
		bitDepth: number;
		/** Memory (GB) the row's quants need at least, e.g. from `minMemoryTierGb`. */
		memGb?: number | null;
		/** The row's quant chips. */
		children: Snippet;
		/** Opens under the row from its label, e.g. a copyable command for the row's main quant. */
		command?: Snippet;
	}

	let { bitDepth, children, command, memGb = null }: Props = $props();

	// collapsed until asked for
	let commandOpen = $state(false);
</script>

{#snippet labelBody()}
	<span class="flex items-center gap-1">
		{#if bitDepth === OTHER_BIT_DEPTH}
			{OTHER_BIT_DEPTH_LABEL}
		{:else}
			{bitDepth}{BIT_DEPTH_LABEL_SUFFIX}
		{/if}

		{#if command}
			<ChevronDown
				class="size-icon-xs shrink-0 transition-transform duration-200 ease-[cubic-bezier(0.23,1,0.32,1)] {commandOpen
					? 'rotate-180'
					: ''}"
			/>
		{/if}
	</span>

	{#if memGb}
		<span class="text-[10px] whitespace-nowrap text-muted-foreground/60 @max-lg:whitespace-normal">
			needs at least {memGb}{GIGABYTE_LABEL} memory
		</span>
	{/if}
{/snippet}

<div class="py-3">
	<div class="grid grid-cols-[5rem_1fr] items-center gap-3 @max-lg:grid-cols-1 @max-lg:gap-1.5">
		{#if command}
			<!-- the row's own label opens its command -->
			<button
				aria-expanded={commandOpen}
				class="flex w-fit cursor-pointer flex-col items-start gap-0.5 text-left text-sm tabular-nums text-muted-foreground transition-colors hover:text-foreground @max-lg:flex-row @max-lg:items-baseline @max-lg:gap-2"
				onclick={() => (commandOpen = !commandOpen)}
				type="button"
			>
				{@render labelBody()}
			</button>
		{:else}
			<span
				class="flex w-fit flex-col items-start gap-0.5 text-sm tabular-nums text-muted-foreground @max-lg:flex-row @max-lg:items-baseline @max-lg:gap-2"
			>
				{@render labelBody()}
			</span>
		{/if}

		<div class="flex flex-wrap justify-end gap-1.5 @max-lg:justify-start">
			{@render children()}
		</div>
	</div>

	{#if command && commandOpen}
		<div transition:slide={{ duration: 200, easing: cubicOut }} class="pt-2">
			{@render command()}
		</div>
	{/if}
</div>
