<script lang="ts">
	import { Database, ScrollText } from '@lucide/svelte';
	import { TruncatedText } from '$lib/components/app';
	import * as Tooltip from '$lib/components/ui/tooltip';
	import { type DraftVariant } from '$lib/constants';
	import {
		CAPABILITY_FLAG_KEYS,
		CAPABILITY_ICONS,
		CAPABILITY_LABELS,
		MODALITY_FLAG_KEYS,
		MODALITY_ICONS,
		MODALITY_LABELS
	} from '$lib/constants';
	import { ModelCapability, ModelModality } from '$lib/enums';
	import { HuggingFaceService } from '$lib/services';
	import { ModelsService } from '$lib/services/models.service';
	import { settingsStore } from '$lib/stores';
	import type { ModelCapabilities, ModelModalities } from '$lib/types/models';
	import { formatParameters } from '$lib/utils';

	interface Props {
		modelId: string;
		hideOrgName?: boolean;
		hideName?: boolean;
		hideModalities?: boolean;
		hideParameters?: boolean;
		showRaw?: boolean;
		showRawTooltip?: boolean;
		hideQuantization?: boolean;
		hideTags?: boolean;
		aliases?: string[];
		tags?: string[];
		/** Render the capability/modality/context icons on a second row. */
		iconsOnNewLine?: boolean;
		modalities?: ModelModalities;
		capabilities?: ModelCapabilities;
		/** Context length in tokens; renders a context icon when set. */
		contextLength?: number;
		/** Min/max GGUF file size (main + draft) across quants; renders a range when set. */
		sizeRange?: { min: number; max: number } | null;
		/** Params badge fallback (formatted) when the model id carries no params token. */
		params?: string;
		draftVariants?: DraftVariant[];
		/** Allow badges to wrap onto new lines instead of truncating. */
		wrap?: boolean;
		class?: string;
	}

	let {
		aliases,
		capabilities,
		class: className = '',
		contextLength,
		draftVariants,
		hideModalities = false,
		hideName = false,
		hideOrgName = false,
		hideParameters = false,
		hideQuantization,
		hideTags,
		iconsOnNewLine = false,
		modalities,
		modelId,
		params,
		showRaw = undefined,
		showRawTooltip = false,
		sizeRange,
		tags,
		wrap = false,
		...rest
	}: Props = $props();

	const badgeClass =
		'inline-flex w-fit shrink-0 items-center justify-center whitespace-nowrap rounded-md border border-border/50 px-1 py-0 text-[10px] font-mono bg-foreground/15 dark:bg-foreground/10 text-foreground [a&]:hover:bg-foreground/25';
	const tagBadgeClass =
		'inline-flex w-fit shrink-0 items-center justify-center whitespace-nowrap rounded-md border border-border/50 px-1 py-0 text-[10px] font-mono text-foreground [a&]:hover:bg-accent [a&]:hover:text-accent-foreground';
	const variantBadgeClass =
		'inline-flex w-fit shrink-0 items-center justify-center whitespace-nowrap rounded-md bg-primary px-1.5 py-0 text-[10px] font-mono font-semibold uppercase tracking-wide text-primary-foreground';

	let parsed = $derived(ModelsService.parseModelId(modelId));
	let resolvedShowRaw = $derived(
		showRaw ?? (settingsStore.config.showRawModelNames as boolean) ?? false
	);
	let resolvedHideQuantization = $derived(
		hideQuantization ?? !settingsStore.config.showModelQuantization
	);
	let resolvedHideTags = $derived(hideTags ?? !settingsStore.config.showModelTags);

	let uniqueAliases = $derived([...new Set(aliases ?? [])]);
	let uniqueTags = $derived([...new Set([...(parsed.tags ?? []), ...(tags ?? [])])]);
	let uniqueDraftVariants = $derived([...new Set(draftVariants ?? [])]);

	const allModalities = [ModelModality.VISION, ModelModality.VIDEO, ModelModality.AUDIO] as const;
	const allCapabilities: ModelCapability[] = [ModelCapability.REASONING, ModelCapability.TOOL_USE];

	let activeModalities = $derived(
		allModalities.filter((modality) => modalities?.[MODALITY_FLAG_KEYS[modality]])
	);
	let activeCapabilities = $derived(
		allCapabilities.filter((capability) => capabilities?.[CAPABILITY_FLAG_KEYS[capability]])
	);

	let primaryAlias = $derived(uniqueAliases.length === 1 ? uniqueAliases[0] : null);
	let displayName = $derived(primaryAlias ?? parsed.modelName ?? modelId);
</script>

{#if resolvedShowRaw}
	<TruncatedText class="font-medium {className}" showTooltip={false} text={modelId} {...rest} />
{:else}
	{#snippet nameAndBadges()}
		{#if !hideName}
			<span class="min-w-0 truncate font-medium">
				{#if !hideOrgName && parsed.orgName}{parsed.orgName}/{/if}{displayName}
			</span>
		{/if}

		<span class="inline-flex items-center gap-1 {wrap ? 'flex-wrap' : ''}">
			{#if parsed.variant}
				<span class={variantBadgeClass} title={`${parsed.variant.toUpperCase()} draft model`}>
					{parsed.variant}
				</span>
			{/if}

			{#if (parsed.params || params) && !hideParameters}
				<span class={badgeClass}>
					{parsed.params ?? params}{parsed.activatedParams ? `-${parsed.activatedParams}` : ''}
				</span>
			{/if}

			{#each uniqueDraftVariants as variant (variant)}
				<span class={variantBadgeClass} title={`${variant.toUpperCase()} draft model available`}>
					{variant}
				</span>
			{/each}

			{#if parsed.quantization && !resolvedHideQuantization}
				<span class={badgeClass}>
					{parsed.quantization}
				</span>
			{/if}

			{#if primaryAlias}
				{#if primaryAlias !== parsed.modelName}
					<span class={badgeClass}>{parsed.modelName ?? modelId}</span>
				{/if}
			{:else if uniqueAliases.length > 1}
				{#each uniqueAliases as alias (alias)}
					<span class={badgeClass}>{alias}</span>
				{/each}
			{/if}

			{#if uniqueTags.length > 0 && !resolvedHideTags}
				{#each uniqueTags as tag (tag)}
					<span class={tagBadgeClass}>{tag}</span>
				{/each}
			{/if}
		</span>
	{/snippet}

	<span
		class="flex min-w-0 items-center gap-1.5 {wrap ? 'flex-wrap' : ''} {iconsOnNewLine
			? 'flex-col items-start'
			: ''} {className}"
		{...rest}
	>
		<span class="flex min-w-0 items-center gap-1.5 {wrap ? 'flex-wrap' : ''}">
			{#if showRawTooltip}
				<Tooltip.Root>
					<Tooltip.Trigger class="flex min-w-0 items-center gap-1.5">
						{@render nameAndBadges()}
					</Tooltip.Trigger>

					<Tooltip.Content>
						<p>{modelId}</p>
					</Tooltip.Content>
				</Tooltip.Root>
			{:else}
				{@render nameAndBadges()}
			{/if}

			{#if activeCapabilities.length > 0 || (activeModalities.length > 0 && !hideModalities)}
				<span class="inline-flex items-center gap-1.25 text-muted-foreground">
					{#each activeCapabilities as capability (capability)}
						{@const CapabilityIcon = CAPABILITY_ICONS[capability]}

						<Tooltip.Root>
							<Tooltip.Trigger>
								<CapabilityIcon class="h-3 w-3 text-muted-foreground" />
							</Tooltip.Trigger>

							<Tooltip.Content>
								<p>{CAPABILITY_LABELS[capability]}</p>
							</Tooltip.Content>
						</Tooltip.Root>
					{/each}

					{#if !hideModalities}
						{#each activeModalities as modality (modality)}
							{@const ModalityIcon = MODALITY_ICONS[modality]}

							<Tooltip.Root>
								<Tooltip.Trigger>
									<ModalityIcon class="h-3 w-3 text-muted-foreground" />
								</Tooltip.Trigger>

								<Tooltip.Content>
									<p>{MODALITY_LABELS[modality]}</p>
								</Tooltip.Content>
							</Tooltip.Root>
						{/each}
					{/if}
				</span>
			{/if}
		</span>

		<span class="inline-flex items-center gap-1.5">
			{#if contextLength}
				<span class="inline-flex items-center gap-1 text-muted-foreground">
					<ScrollText class="h-3 w-3" />

					<span class="text-xs">{formatParameters(contextLength)}</span>
				</span>
			{/if}

			{#if sizeRange}
				<span class="inline-flex items-center gap-1 text-muted-foreground">
					<Database class="h-3 w-3" />

					<span class="text-xs"
						>{HuggingFaceService.formatSizeRange(sizeRange.min, sizeRange.max)}</span
					>
				</span>
			{/if}
		</span>
	</span>
{/if}
