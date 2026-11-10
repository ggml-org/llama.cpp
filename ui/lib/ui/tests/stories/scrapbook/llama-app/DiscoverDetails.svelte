<!--
	A repo's details in Llama Chat's Discover, drawn with Model UI the way ModelsDiscoverDetails draws them: the header
	(base model's avatar with the repo's, the repo id with its capability icons, the base model, View on Hugging Face),
	the Hub's stats, the metadata chips, then the repo's quants in the quant panel as download chips.
-->
<script lang="ts">
	import type { Build, Family } from './catalog';
	import { bitDepthOf, memFor } from './catalog-helpers';
	import { formatDownloads, type HubModel, updatedAgo } from './hub';
	import { Download, ExternalLink, Heart, MessageSquareCode } from '@lucide/svelte';
	import { Button } from '$lib/components/ui/button';
	import {
		ModelCapabilityIcons,
		ModelMetadataItem,
		ModelOrgAvatar,
		ModelQuantChip,
		ModelQuantOptions,
		ModelQuantRow
	} from '$lib/components/models';
	import { formatParameters } from '$lib/models';

	let { family, model }: { family: Family; model: HubModel } = $props();

	let org = $derived(model.id.split('/')[0]);
	let baseOrg = $derived(model.baseModel?.split('/')[0] ?? org);
	let builds = $derived(family.sizes.flatMap((size) => size.builds).filter((build) => build.repo === model.id));
	let rows = $derived.by(() => {
		const byDepth = new Map<number, Build[]>();

		for (const build of builds) {
			const depth = bitDepthOf(build.quant);

			byDepth.set(depth, [...(byDepth.get(depth) ?? []), build]);
		}

		return [...byDepth].sort(([a], [b]) => b - a).map(([bitDepth, list]) => ({ bitDepth, builds: list }));
	});
</script>

<div class="space-y-6 p-6">
	<header class="space-y-3">
		<div class="flex flex-wrap items-start justify-between gap-3">
			<div class="flex min-w-0 items-center gap-2">
				<ModelOrgAvatar
					org={baseOrg}
					quantOrg={baseOrg !== org ? org : undefined}
					quantPositionClass="-bottom-1.5 -right-1.5"
					quantSize="size-6"
					size="size-12"
				/>

				<div class="min-w-0">
					<div class="flex items-center gap-2 @max-lg:flex-col @max-lg:items-start">
						<h1 class="truncate text-lg font-semibold @max-lg:w-full">{model.id}</h1>

						<ModelCapabilityIcons
							gapClass="gap-2"
							iconSize="size-icon-md"
							modalities={{ audio: model.audio, video: model.video, vision: model.vision }}
							supportsThinking={model.reasoning}
							supportsToolUse={model.tools}
						/>
					</div>

					{#if model.baseModel}
						<div class="flex items-center gap-1">
							<span class="truncate text-xs text-muted-foreground">{model.baseModel}</span>
							<ExternalLink class="size-icon-xs shrink-0 text-muted-foreground" />
						</div>
					{/if}
				</div>
			</div>

			<Button class="h-auto gap-1.5 px-2.5 py-1.5 text-xs" size="sm" variant="secondary">
				View on Hugging Face
				<ExternalLink class="size-3.5" />
			</Button>
		</div>

		<div class="flex flex-wrap items-center gap-x-4 gap-y-1 text-sm text-muted-foreground">
			<span class="inline-flex items-center gap-1.5"><Download class="size-icon-sm" />{formatDownloads(model.downloads)}</span>
			<span class="inline-flex items-center gap-1.5"><Heart class="size-icon-sm" />{model.likes}</span>
			<span>Updated {updatedAgo(model.lastModified)}</span>
		</div>

		<p class="text-sm text-muted-foreground">{family.description.replaceAll('**', '')}</p>

		<div class="flex flex-wrap items-center gap-1.5">
			{#if model.params}<ModelMetadataItem label="Model size" value="{formatParameters(model.params)} params" />{/if}
			{#if model.contextLength}<ModelMetadataItem label="Context" value={model.contextLength.toLocaleString('en')} />{/if}
			{#if model.architecture}<ModelMetadataItem label="Architecture" value={model.architecture} />{/if}

			<Button class="h-auto gap-1.5 px-2.5 py-1 text-xs" size="sm" variant="secondary">
				<MessageSquareCode class="size-3" />
				Chat template
			</Button>

			{#if model.license}<ModelMetadataItem label="License" value={model.license} />{/if}
		</div>
	</header>

	<ModelQuantOptions>
		{#each rows as row (row.bitDepth)}
			<ModelQuantRow bitDepth={row.bitDepth} memGb={memFor(row.builds[0])}>
				{#each row.builds as build (build.quant)}
					<ModelQuantChip label={build.quant?.toUpperCase() ?? 'GGUF'} sizeBytes={build.sizeBytes} />
				{/each}

				{#snippet command()}
					<pre
						class="overflow-x-auto rounded-panel bg-(--surface-code) px-3 py-2 text-xs text-(--fg-code)">llama serve -hf {model.id}:{row.builds[0].quant}</pre>
				{/snippet}
			</ModelQuantRow>
		{/each}
	</ModelQuantOptions>
</div>
