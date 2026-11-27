<!-- One way through a phone sheet: what opens it, then its pages in the order a tap pushes them -->
<script lang="ts">
	import type { Flow } from './flows';
	import { ChevronRight } from '@lucide/svelte';

	let { flow }: { flow: Flow } = $props();
</script>

<div class="flex flex-col gap-3">
	<p class="text-caption text-muted-foreground">{flow.trigger}</p>

	<ol class="flex flex-wrap items-stretch gap-2">
		{#each flow.pages as page, index (page.title)}
			{#if index > 0}
				<li aria-hidden="true" class="flex items-center text-muted-foreground">
					<ChevronRight class="size-4" />
				</li>
			{/if}

			<li class="inset-group-surface flex w-40 flex-col gap-1 px-3 py-2.5">
				<span class="text-label">{page.title}</span>

				<span class="text-caption text-muted-foreground">{page.note}</span>
			</li>
		{/each}
	</ol>
</div>
