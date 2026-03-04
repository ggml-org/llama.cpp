<script lang="ts">
	import { ModelsService } from '$lib/services/models.service';
	import { config } from '$lib/stores/settings.svelte';

	interface Props {
		modelId: string;
		showOrgName?: boolean;
		showRaw?: boolean;
		aliases?: string[];
		tags?: string[];
		class?: string;
	}

	let {
		modelId,
		showOrgName = false,
		showRaw = undefined,
		aliases,
		tags,
		class: className = ''
	}: Props = $props();

	const badgeClass =
		'inline-flex w-fit shrink-0 items-center justify-center whitespace-nowrap rounded-md border border-border/50 px-1 py-0 text-[10px] font-mono bg-foreground/15 dark:bg-foreground/10 text-foreground [a&]:hover:bg-foreground/25';
	const tagBadgeClass =
		'inline-flex w-fit shrink-0 items-center justify-center whitespace-nowrap rounded-md border border-border/50 px-1 py-0 text-[10px] font-mono text-foreground [a&]:hover:bg-accent [a&]:hover:text-accent-foreground';

	let parsed = $derived(ModelsService.parseModelId(modelId));
	let resolvedShowRaw = $derived(showRaw ?? (config().showRawModelNames as boolean) ?? false);
</script>

{#if resolvedShowRaw}
	<span class="min-w-0 truncate font-medium {className}">{modelId}</span>
{:else}
	<span class="flex min-w-0 flex-wrap items-center gap-1 {className}">
		<span class="min-w-0 truncate font-medium">
			{#if showOrgName}{parsed.orgName}/{/if}{parsed.modelName ?? modelId}
		</span>

		{#if parsed.params}
			<span class={badgeClass}>
				{parsed.params}{parsed.activatedParams ? `-${parsed.activatedParams}` : ''}
			</span>
		{/if}

		{#if parsed.quantization}
			<span class={badgeClass}>
				{parsed.quantization}
			</span>
		{/if}

		{#if aliases && aliases.length > 0}
			{#each aliases as alias (alias)}
				<span class={badgeClass}>{alias}</span>
			{/each}
		{/if}

		{#if tags && tags.length > 0}
			{#each tags as tag (tag)}
				<span class={tagBadgeClass}>{tag}</span>
			{/each}
		{/if}
	</span>
{/if}
