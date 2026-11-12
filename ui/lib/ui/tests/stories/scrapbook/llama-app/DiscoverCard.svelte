<!--
	A repo in Llama Chat's Discover, drawn with Model UI the way ModelsDiscoverCard draws it (allozaur/feat/models-manager-
	redesign): the base model's org with the repo's in the corner, the name stacked over its badges, the capabilities
	Discover works out as labelled badges, the catalog's description, the download size and context at the foot and the
	Hub's download count at the end.
-->
<script lang="ts">
	import type { Family } from './catalog';
	import { formatDownloads, type HubModel } from './hub';
	import { Database, Download, ScrollText } from '@lucide/svelte';
	import { ModelCapabilityIcons, ModelCard, ModelId, ModelOrgAvatar } from '$lib/components/models';
	import { formatFileSize, formatParameters, formatSizeRange } from '$lib/models';

	let { family, model }: { family: Family; model: HubModel } = $props();

	let org = $derived(model.id.split('/')[0]);
	let baseOrg = $derived(model.baseModel?.split('/')[0] ?? org);
	let size = $derived(family.sizes.find((s) => s.builds.some((build) => build.repo === model.id)));
	let bytes = $derived((size?.builds ?? []).filter((b) => b.repo === model.id).map((b) => b.sizeBytes ?? 0));
</script>

<ModelCard onSelect={() => {}} tooltip={model.id}>
	{#snippet avatar()}
		<ModelOrgAvatar org={baseOrg} quantOrg={org} />
	{/snippet}

	{#snippet title()}
		<ModelId hideOrgName name={model.id.split('/')[1]} params={size?.params} stackId />
	{/snippet}

	<ModelCapabilityIcons
		asBadges
		modalities={{ audio: model.audio, video: model.video, vision: model.vision }}
		supportsThinking={model.reasoning}
		supportsToolUse={model.tools}
	/>

	<p class="line-clamp-2 text-xs text-muted-foreground">{family.description.replaceAll('**', '')}</p>

	{#snippet footer()}
		<span class="flex items-center gap-3 whitespace-nowrap">
			{#if bytes.length}
				<span class="inline-flex items-center gap-1" title="Download size">
					<Database class="size-icon-xs" />

					{bytes.length === 1
						? formatFileSize(bytes[0])
						: formatSizeRange(Math.min(...bytes), Math.max(...bytes))}
				</span>
			{/if}

			{#if model.contextLength}
				<span class="inline-flex items-center gap-1" title="Context length">
					<ScrollText class="size-icon-xs" />

					{formatParameters(model.contextLength)}
				</span>
			{/if}
		</span>
	{/snippet}

	{#snippet footerEnd()}
		<span class="inline-flex items-center gap-1 text-muted-foreground" title="Downloads">
			<Download class="size-icon-xs" />

			{formatDownloads(model.downloads)}
		</span>
	{/snippet}
</ModelCard>
