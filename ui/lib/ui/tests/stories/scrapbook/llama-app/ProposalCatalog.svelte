<!--
	llama.app's models page, proposed: the catalog's sections as Discover's grid of model cards (MODEL_CARD_GRID_CLASS
	in ui/apps/chat), each section headed the way the models page heads its own, with its count beside the title.
-->
<script lang="ts">
	import type { Family } from './catalog';
	import { releasedFor } from './catalog-helpers';
	import LlamaAppModelCard from './LlamaAppModelCard.svelte';
	import { ArrowUpRight } from '@lucide/svelte';

	let { families }: { families: Family[] } = $props();

	let chat = $derived(families.filter((family) => !family.decision && !family.embedding));
	let sections = $derived([
		{ families: chat.filter((family) => family.featured), title: 'Good places to start' },
		{ families: chat.filter((family) => !family.featured), title: 'More models' },
		{
			families: families.filter((family) => family.decision),
			lead: 'They take a document and typed questions, and return a probability for each option. They don’t chat: apps call them through the API.',
			title: 'Decision models'
		},
		{
			families: families.filter((family) => family.embedding),
			lead: 'They turn text, and for some images and audio, into vectors for search and retrieval. They don’t chat: apps call them through the API.',
			title: 'Embedding models'
		}
	]);
	let newest = $derived([...families].sort((a, b) => b.released.localeCompare(a.released))[0]);
</script>

<main class="flex flex-col gap-10 px-5 py-8 @3xl:px-10 @3xl:py-12">
	<header class="flex max-w-2xl flex-col gap-3">
		<h1 class="text-[2.25rem] leading-[1.05] font-bold tracking-[-0.025em] @3xl:text-[3rem]">Models</h1>

		<p class="text-body text-muted-foreground">
			Open models picked to run well in Llama, on the computer you already have.
			<span class="tabular-nums">Updated {releasedFor(newest)}.</span>
		</p>
	</header>

	{#each sections as section (section.title)}
		{#if section.families.length}
			<section class="flex flex-col gap-3">
				<div class="flex flex-col gap-1">
					<h2 class="flex items-baseline gap-2 text-title">
						{section.title}
						<span class="text-label font-normal text-muted-foreground tabular-nums">{section.families.length}</span>
					</h2>

					{#if section.lead}
						<p class="max-w-2xl text-body-sm text-muted-foreground">{section.lead}</p>
					{/if}
				</div>

				<div class="grid grid-cols-[repeat(auto-fill,minmax(18rem,1fr))] gap-3">
					{#each section.families as family (family.name)}
						<LlamaAppModelCard {family} href="#{family.name}" />
					{/each}
				</div>
			</section>
		{/if}
	{/each}

	<p class="text-body-sm text-muted-foreground">
		Thousands more GGUF models are on
		<a
			class="inline-flex items-center gap-0.5 font-medium text-foreground underline decoration-border underline-offset-3"
			href="https://huggingface.co/models?library=gguf"
		>
			Hugging Face<ArrowUpRight class="size-icon-xs" />
		</a>
	</p>
</main>
