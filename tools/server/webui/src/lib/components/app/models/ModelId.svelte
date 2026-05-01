<script lang="ts">
	import { ModelsService } from '$lib/services/models.service';
	import { config } from '$lib/stores/settings.svelte';
	import { TruncatedText } from '$lib/components/app';

	interface Props {
		modelId: string;
		hideOrgName?: boolean;
		showRaw?: boolean;
		hideQuantization?: boolean;
		aliases?: string[];
		tags?: string[];
		class?: string;
	}

	let {
		modelId,
		hideOrgName = false,
		showRaw = undefined,
		hideQuantization = false,
		aliases,
		tags,
		class: className = '',
		...rest
	}: Props = $props();

	const badgeClass =
		'inline-flex w-fit shrink-0 items-center justify-center whitespace-nowrap rounded-md border border-border/50 px-1 py-0 text-[10px] font-mono bg-foreground/15 dark:bg-foreground/10 text-foreground [a&]:hover:bg-foreground/25';
	const tagBadgeClass =
		'inline-flex w-fit shrink-0 items-center justify-center whitespace-nowrap rounded-md border border-border/50 px-1 py-0 text-[10px] font-mono text-foreground [a&]:hover:bg-accent [a&]:hover:text-accent-foreground';

	let parsed = $derived(ModelsService.parseModelId(modelId));
	let resolvedShowRaw = $derived(showRaw ?? (config().showRawModelNames as boolean) ?? false);
	let displayName = $derived(
		aliases && aliases.length > 0 ? aliases[0] : (parsed.modelName ?? modelId)
	);
	let remainingAliases = $derived(aliases && aliases.length > 1 ? aliases.slice(1) : []);
	let allTags = $derived([...(parsed.tags ?? []), ...(tags ?? [])]);
</script>

{#if resolvedShowRaw}
	<TruncatedText class="font-medium {className}" showTooltip={false} text={modelId} {...rest} />
{:else}
	<span class="flex min-w-0 flex-wrap items-center gap-1 {className}" {...rest}>
		<span class="min-w-0 truncate font-medium">
			{#if !hideOrgName && parsed.orgName && !(aliases && aliases.length > 0)}{parsed.orgName}/{/if}{displayName}
		</span>

		{#if parsed.params}
			<span class={badgeClass}>
				{parsed.params}{parsed.activatedParams ? `-${parsed.activatedParams}` : ''}
			</span>
		{/if}

		{#if parsed.quantization && !hideQuantization}
			<span class={badgeClass}>
				{parsed.quantization}
			</span>
		{/if}

		{#if remainingAliases.length > 0}
			{#each remainingAliases as alias (alias)}
				<span class={badgeClass}>{alias}</span>
			{/each}
		{/if}

		{#if allTags.length > 0}
			{#each allTags as tag (tag)}
				<span class={tagBadgeClass}>{tag}</span>
			{/each}
		{/if}
	</span>
{/if}
