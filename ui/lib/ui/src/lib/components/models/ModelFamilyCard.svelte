<!--
	A model family as llama.app's catalog shows it, built on ModelCard: the maker's avatar with the publisher of the
	GGUFs in the corner, the name over its sizes in one quiet line, what it can do as icons at the top right (each named
	in its tooltip), the description, then the download and the memory it needs at the foot and the release month (or
	Non-commercial, or a status such as "Downloaded") at the end. The whole card is one link or button.
-->
<script lang="ts">
	import ModelCapabilityIcons from './ModelCapabilityIcons.svelte';
	import ModelCard from './ModelCard.svelte';
	import ModelOrgAvatar from './ModelOrgAvatar.svelte';
	import { Database, MemoryStick } from '@lucide/svelte';
	import { formatFileSize, formatSizeRange } from '../../models/format';
	import type { ModelModalities, ModelSizeRange } from '../../models/types';
	import type { Snippet } from 'svelte';

	interface Props {
		/** The family's name, e.g. `Gemma 4`. */
		name: string;
		/** The maker's Hugging Face org, e.g. `google`; its avatar leads the card. */
		org: string;
		/** Who published the GGUFs, e.g. `ggml-org`, in the avatar's corner when it isn't the maker. */
		quantOrg?: string;
		/** The family's sizes in one line, e.g. from `summarizeSizes`: `E2B–31B · 5 sizes`. */
		sizes: string;
		modalities?: ModelModalities;
		supportsThinking?: boolean;
		supportsToolUse?: boolean;
		/** Download size across the family's files, in bytes. */
		sizeRange?: ModelSizeRange | null;
		/** Memory (GB) the smallest size needs. */
		memGb?: number | null;
		/** When it came out, e.g. `Mar 2026`; shown at the end of the foot. */
		released?: string;
		/** The license forbids commercial use; takes the release month's place, in the warning ink. */
		nonCommercial?: boolean;
		/** Takes the foot's end before Non-commercial and the release month, e.g. a status Badge saying the family is
		 * already on this computer. */
		status?: Snippet;
		/** Plain description, two lines at most; `children` replaces it for richer text. */
		description?: string;
		children?: Snippet;
		href?: string;
		onSelect?: () => void;
		class?: string;
	}

	let {
		children,
		class: className = '',
		description,
		href,
		memGb = null,
		modalities,
		name,
		nonCommercial = false,
		onSelect,
		org,
		quantOrg,
		released,
		sizeRange = null,
		sizes,
		status,
		supportsThinking = false,
		supportsToolUse = false
	}: Props = $props();
</script>

<ModelCard class={className} {href} {onSelect} tooltip={name}>
	{#snippet avatar()}
		<ModelOrgAvatar {org} {quantOrg} />
	{/snippet}

	{#snippet title()}
		<span class="flex min-w-0 flex-col gap-0.5">
			<span class="truncate font-medium">{name}</span>

			<span class="truncate font-mono text-caption text-muted-foreground tabular-nums">{sizes}</span>
		</span>
	{/snippet}

	<!-- at the top right, where the card keeps its controls: above the card's link, so each icon's tooltip is its own
	     button rather than one nested in the link -->
	{#snippet actions()}
		<span class="flex h-9 items-center pr-1.5">
			<ModelCapabilityIcons {modalities} {supportsThinking} {supportsToolUse} />
		</span>
	{/snippet}

	<!-- body-sm rather than the app cards' 12px: on a page people read to choose, the two lines carry the pitch -->
	<div class="line-clamp-2 text-body-sm text-pretty text-muted-foreground">
		{#if children}
			{@render children()}
		{:else if description}
			{description}
		{/if}
	</div>

	{#snippet footer()}
		<span class="flex items-center gap-3 whitespace-nowrap tabular-nums">
			{#if sizeRange}
				<span class="inline-flex items-center gap-1" title="Download size">
					<Database class="size-icon-xs" />

					{sizeRange.min === sizeRange.max
						? formatFileSize(sizeRange.min)
						: formatSizeRange(sizeRange.min, sizeRange.max)}
				</span>
			{/if}

			{#if memGb}
				<span class="inline-flex items-center gap-1" title="Memory the smallest size needs">
					<MemoryStick class="size-icon-xs" />

					{memGb} GB+
				</span>
			{/if}
		</span>
	{/snippet}

	{#snippet footerEnd()}
		{#if status}
			{@render status()}
		{:else if nonCommercial}
			<!-- in the warning ink rather than a badge, so it costs no more room than the date it replaces -->
			<span class="text-(--fg-warning)" title="The license forbids commercial use">Non-commercial</span>
		{:else if released}
			<span class="text-muted-foreground tabular-nums" title="Released">{released}</span>
		{/if}
	{/snippet}
</ModelCard>
