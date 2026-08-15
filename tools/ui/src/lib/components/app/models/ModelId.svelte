<script lang="ts">
	import { TruncatedText } from '$lib/components/app';
	import { Image, Lightbulb, Mic, Video } from '@lucide/svelte';
	import * as Tooltip from '$lib/components/ui/tooltip';
	import { ModelsService } from '$lib/services/models.service';
	import { settingsStore } from '$lib/stores';
	import type { ModelModalities } from '$lib/types/models';

	interface Props {
		modelId: string;
		hideOrgName?: boolean;
		showRaw?: boolean;
		showRawTooltip?: boolean;
		hideQuantization?: boolean;
		hideTags?: boolean;
		aliases?: string[];
		tags?: string[];
		modalities?: ModelModalities;
		supportsThinking?: boolean;
		class?: string;
	}

	let {
		aliases,
		class: className = '',
		hideOrgName = false,
		hideQuantization,
		hideTags,
		modalities,
		modelId,
		showRaw = undefined,
		showRawTooltip = false,
		supportsThinking = false,
		tags,
		...rest
	}: Props = $props();

	const badgeClass =
		'inline-flex w-fit shrink-0 items-center justify-center whitespace-nowrap rounded-md border border-border/50 px-1 py-0 text-[10px] font-mono bg-foreground/15 dark:bg-foreground/10 text-foreground [a&]:hover:bg-foreground/25';
	const tagBadgeClass =
		'inline-flex w-fit shrink-0 items-center justify-center whitespace-nowrap rounded-md border border-border/50 px-1 py-0 text-[10px] font-mono text-foreground [a&]:hover:bg-accent [a&]:hover:text-accent-foreground';

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
	let hasModalityIcons = $derived(
		supportsThinking || modalities?.vision || modalities?.video || modalities?.audio
	);

	let primaryAlias = $derived(uniqueAliases.length === 1 ? uniqueAliases[0] : null);
	let displayName = $derived(primaryAlias ?? parsed.modelName ?? modelId);
</script>

{#if resolvedShowRaw}
	<TruncatedText class="font-medium {className}" showTooltip={false} text={modelId} {...rest} />
{:else}
	{#snippet nameAndBadges()}
		<span class="min-w-0 truncate font-medium">
			{#if !hideOrgName && parsed.orgName}{parsed.orgName}/{/if}{displayName}
		</span>

		<span class="inline-flex items-center gap-1">
			{#if parsed.params}
				<span class={badgeClass}>
					{parsed.params}{parsed.activatedParams ? `-${parsed.activatedParams}` : ''}
				</span>
			{/if}

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

	<span class="flex min-w-0 items-center gap-1.5 {className}" {...rest}>
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

		{#if hasModalityIcons}
			<span class="inline-flex items-center gap-1.25 text-muted-foreground">
				{#if supportsThinking}
					<Tooltip.Root>
						<Tooltip.Trigger>
							<Lightbulb class="h-3 w-3 text-muted-foreground" />
						</Tooltip.Trigger>
						<Tooltip.Content>
							<p>Reasoning</p>
						</Tooltip.Content>
					</Tooltip.Root>
				{/if}

				{#if modalities?.vision}
					<Tooltip.Root>
						<Tooltip.Trigger>
							<Image class="h-3 w-3 text-muted-foreground" />
						</Tooltip.Trigger>
						<Tooltip.Content>
							<p>Vision</p>
						</Tooltip.Content>
					</Tooltip.Root>
				{/if}

				{#if modalities?.video}
					<Tooltip.Root>
						<Tooltip.Trigger>
							<Video class="h-3 w-3 text-muted-foreground" />
						</Tooltip.Trigger>
						<Tooltip.Content>
							<p>Video</p>
						</Tooltip.Content>
					</Tooltip.Root>
				{/if}

				{#if modalities?.audio}
					<Tooltip.Root>
						<Tooltip.Trigger>
							<Mic class="h-3 w-3 text-muted-foreground" />
						</Tooltip.Trigger>
						<Tooltip.Content>
							<p>Audio</p>
						</Tooltip.Content>
					</Tooltip.Root>
				{/if}
			</span>
		{/if}
	</span>
{/if}
