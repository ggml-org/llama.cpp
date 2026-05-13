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

	let uniqueAliases = $derived([...new Set(aliases ?? [])]);
	let uniqueTags = $derived([...new Set([...(parsed.tags ?? []), ...(tags ?? [])])]);

	let primaryAlias = $derived(uniqueAliases.length === 1 ? uniqueAliases[0] : null);
	let displayName = $derived(primaryAlias ?? parsed.modelName ?? modelId);
</script>

{#if resolvedShowRaw}
	<TruncatedText class="font-medium {className}" showTooltip={false} text={modelId} {...rest} />
{:else}
	<span class="flex min-w-0 flex-wrap items-center gap-1 {className}" {...rest}>
		<span class="min-w-0 truncate font-medium">
			{#if !hideOrgName && parsed.orgName}{parsed.orgName}/{/if}{displayName}
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

		{#if primaryAlias}
			{#if primaryAlias !== parsed.modelName}
				<span class={badgeClass}>{parsed.modelName ?? modelId}</span>
			{/if}
		{:else if uniqueAliases.length > 1}
			{#each uniqueAliases as alias (alias)}
				<span class={badgeClass}>{alias}</span>
			{/each}
		{/if}

		{#if uniqueTags.length > 0}
			{#each uniqueTags as tag (tag)}
				<span class={tagBadgeClass}>{tag}</span>
			{/each}
		{/if}
	</span>
{/if}
