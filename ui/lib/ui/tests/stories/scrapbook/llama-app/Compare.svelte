<!--
	Llama Chat's Discover beside llama.app's catalog: the same models, each side in a size container of its own, with
	what differs listed above them.
-->
<script lang="ts">
	import type { Snippet } from 'svelte';

	interface Props {
		app: Snippet;
		site: Snippet;
		/** What differs between the two, one line each. */
		notes?: string[];
	}

	let { app, notes = [], site }: Props = $props();
</script>

<div class="flex min-h-screen flex-col gap-6 bg-background p-6">
	{#if notes.length}
		<ul class="max-w-3xl list-disc space-y-1 pl-5 text-body-sm text-muted-foreground marker:text-muted-foreground/60">
			{#each notes as note (note)}
				<li>{note}</li>
			{/each}
		</ul>
	{/if}

	<div class="grid grid-cols-1 items-start gap-6 lg:grid-cols-2">
		{#each [{ body: app, label: 'Llama Chat · Discover', sub: 'a repo on Hugging Face' }, { body: site, label: 'llama.app · catalog', sub: 'a family from the catalog' }] as side (side.label)}
			<section class="min-w-0">
				<p class="mb-3 flex items-baseline gap-2">
					<span class="font-mono text-label-sm tracking-[0.12em] uppercase">{side.label}</span>
					<span class="text-caption text-muted-foreground">{side.sub}</span>
				</p>

				<div class="@container overflow-hidden rounded-card bg-background shadow-[0_0_0_1px_var(--line-default)]">
					{@render side.body()}
				</div>
			</section>
		{/each}
	</div>
</div>
