<script lang="ts">
	import ModelCapabilityIcons from './ModelCapabilityIcons.svelte';
	import type { ModelDraft } from './ModelsManager/utils';
	import { Database, ScrollText } from '@lucide/svelte';
	import { TruncatedText } from '$lib/components/app';
	import * as Tooltip from '$lib/components/ui/tooltip';
	import { type ModelSidecar } from '$lib/constants';
	import { MODEL_BADGE_CLASS, MODEL_VARIANT_BADGE_CLASS } from '$lib/constants';
	import { HuggingFaceService } from '$lib/services';
	import { ModelsService } from '$lib/services/models.service';
	import { settingsStore } from '$lib/stores';
	import type { ModelModalities } from '$lib/types/models';
	import { isAuxSidecar } from '$lib/utils';
	import { formatParameters } from '$lib/utils';

	interface Props {
		modelId: string;
		hideOrgName?: boolean;
		hideName?: boolean;
		hideModalities?: boolean;
		hideCapabilities?: boolean;
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
		supportsThinking?: boolean;
		supportsToolUse?: boolean;
		/** Native title for the root element; keeps long lists light where a floating tooltip per row is too costly. */
		title?: string;
		/** Context length in tokens; renders a context icon when set. */
		contextLength?: number;
		/** Min/max GGUF file size (main + draft) across quants; renders a range when set. */
		sizeRange?: { min: number; max: number } | null;
		draftSidecars?: ModelSidecar[];
		/** Drafts a load would use, rendered as `+ [KIND] [QUANT]` next to the id. */
		drafts?: ModelDraft[];
		/** Allow badges to wrap onto new lines instead of truncating. */
		wrap?: boolean;
		class?: string;
	}

	let {
		aliases,
		class: className = '',
		contextLength,
		drafts = [],
		draftSidecars = [],
		hideCapabilities = false,
		hideModalities = false,
		hideName = false,
		hideOrgName = false,
		hideParameters = false,
		hideQuantization,
		hideTags,
		iconsOnNewLine = false,
		modalities,
		modelId,
		showRaw = undefined,
		showRawTooltip = false,
		sizeRange,
		supportsThinking = false,
		supportsToolUse = false,
		tags,
		title,
		wrap = false,
		...rest
	}: Props = $props();

	const badgeClass = MODEL_BADGE_CLASS;
	const tagBadgeClass =
		'inline-flex w-fit shrink-0 items-center justify-center whitespace-nowrap rounded-md border border-border/50 px-1 py-0 text-[10px] font-mono text-foreground [a&]:hover:bg-accent [a&]:hover:text-accent-foreground';
	const variantBadgeClass = MODEL_VARIANT_BADGE_CLASS;

	/** Alias badges beyond this many collapse into a single `+x more` badge. */
	const MAX_ALIAS_BADGES = 2;

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
	let uniqueDraftSidecars = $derived([...new Set(draftSidecars)].filter((s) => !isAuxSidecar(s)));
	let activeDrafts = $derived(drafts.filter((draft) => draft.active && draft.kind));

	let primaryAlias = $derived(uniqueAliases.length === 1 ? uniqueAliases[0] : null);
	let displayName = $derived(primaryAlias ?? parsed.modelName ?? modelId);

	let hasBadges = $derived(
		parsed.sidecar ||
			(parsed.params && !hideParameters) ||
			(parsed.quantization && !resolvedHideQuantization) ||
			primaryAlias ||
			uniqueAliases.length > 1 ||
			(uniqueTags.length > 0 && !resolvedHideTags)
	);
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

		{#if hasBadges}
			<!-- the badges do not shrink, so the group has to clip them: without this
			     they paint over the row actions when the row is narrow -->
			<span
				class="inline-flex min-w-0 items-center gap-1 overflow-hidden {wrap ? 'flex-wrap' : ''}"
			>
				{#if parsed.sidecar}
					<span class={variantBadgeClass} title={`${parsed.sidecar.toUpperCase()} draft model`}>
						{parsed.sidecar}
					</span>
				{/if}

				{#if parsed.params && !hideParameters}
					<span class={badgeClass}>
						{parsed.params}{parsed.activatedParams ? `-${parsed.activatedParams}` : ''}
					</span>
				{/if}

				{#if parsed.quantization && !resolvedHideQuantization}
					<span class={badgeClass}>
						{parsed.quantization}
					</span>
				{/if}

				{#each activeDrafts as draft (draft.kind ?? 'draft')}
					<span class="flex shrink-0 items-center gap-1">
						<span class="text-[10px] text-muted-foreground">+</span>

						<span class={variantBadgeClass} title="Speculative draft in use">{draft.kind}</span>

						{#if draft.quant}
							<span class={badgeClass}>{draft.quant}</span>
						{/if}
					</span>
				{/each}

				{#each uniqueDraftSidecars as sidecar (sidecar)}
					<span class={variantBadgeClass} title={`${sidecar.toUpperCase()} draft model available`}>
						{sidecar}
					</span>
				{/each}

				{#if primaryAlias}
					{#if primaryAlias !== parsed.modelName}
						<span class="{badgeClass} max-w-32 truncate" title={parsed.modelName ?? modelId}>
							{parsed.modelName ?? modelId}
						</span>
					{/if}
				{:else if uniqueAliases.length > 1}
					{#each uniqueAliases.slice(0, MAX_ALIAS_BADGES) as alias (alias)}
						<span class="{badgeClass} max-w-32 truncate" title={alias}>
							{alias}
						</span>
					{/each}

					{#if uniqueAliases.length > MAX_ALIAS_BADGES}
						<span class={badgeClass} title={uniqueAliases.slice(MAX_ALIAS_BADGES).join(', ')}>
							+{uniqueAliases.length - MAX_ALIAS_BADGES} more
						</span>
					{/if}
				{/if}

				{#if uniqueTags.length > 0 && !resolvedHideTags}
					{#each uniqueTags as tag (tag)}
						<span class={tagBadgeClass}>{tag}</span>
					{/each}
				{/if}
			</span>
		{/if}
	{/snippet}

	<!-- overflow-hidden bounds the whole id: the badges, tags and capability icons do
	     not shrink, so without it they spill past the row and over its actions -->
	<span
		class="flex min-w-0 items-center gap-1.5 overflow-hidden {wrap
			? 'flex-wrap'
			: ''} {iconsOnNewLine ? 'flex-col items-start' : ''} {className}"
		{title}
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

			{#if !iconsOnNewLine}
				<ModelCapabilityIcons
					{hideCapabilities}
					{hideModalities}
					{modalities}
					{supportsThinking}
					{supportsToolUse}
				/>
			{/if}
		</span>

		{#if iconsOnNewLine || contextLength || sizeRange}
			<span class="inline-flex items-center gap-1.5">
				{#if iconsOnNewLine}
					<ModelCapabilityIcons
						{hideCapabilities}
						{hideModalities}
						{modalities}
						{supportsThinking}
						{supportsToolUse}
					/>
				{/if}

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
		{/if}
	</span>
{/if}
