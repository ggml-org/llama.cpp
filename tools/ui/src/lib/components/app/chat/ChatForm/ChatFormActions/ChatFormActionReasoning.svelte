<script lang="ts">
	import { Check, Info, Lightbulb, LightbulbOff } from '@lucide/svelte';
	import { Button } from '$lib/components/ui/button';
	import * as DropdownMenu from '$lib/components/ui/dropdown-menu';
	import * as Tooltip from '$lib/components/ui/tooltip';
	import { ICON_CLASS_DEFAULT } from '$lib/constants';
	import { ReasoningEffort } from '$lib/enums';
	import { useReasoningMenu } from '$lib/hooks/use-reasoning-menu.svelte';

	const reasoning = useReasoningMenu();

	// "Default" is the resting state, where the bulb alone carries the meaning
	let isDefault = $derived(reasoning.currentEffort === ReasoningEffort.DEFAULT);
</script>

<!-- the level belongs to the chat, not to the model list -->
<DropdownMenu.Root>
	<DropdownMenu.Trigger>
		{#snippet child({ props })}
			<Button
				{...props}
				aria-label="Reasoning effort"
				class="h-auto gap-1 rounded-sm {isDefault ? 'px-1' : 'px-1.75!'} py-1 text-xs"
				variant="ghost"
			>
				<span class="flex items-center gap-0.75 {reasoning.isOff ? 'text-muted-foreground' : ''}">
					{#if reasoning.isOff}
						<LightbulbOff class="size-3 shrink-0" />
					{:else}
						<Lightbulb class="size-3 shrink-0" />
					{/if}

					{#if !isDefault}
						<span class="capitalize">{reasoning.currentEffort}</span>
					{/if}
				</span>
			</Button>
		{/snippet}
	</DropdownMenu.Trigger>

	<DropdownMenu.Content align="end" class="min-w-56">
		{#each reasoning.levels as level (level.value)}
			{@const tokenLabel = reasoning.tokenLabel(level)}

			<DropdownMenu.Item class="gap-3" onSelect={() => reasoning.select(level)}>
				{#if reasoning.isSelected(level)}
					<Check class="{ICON_CLASS_DEFAULT} shrink-0 text-foreground" />
				{:else}
					<div class="{ICON_CLASS_DEFAULT} shrink-0"></div>
				{/if}

				<span class="min-w-0 flex-1 truncate">{level.label}</span>

				{#if tokenLabel}
					<span class="shrink-0 text-[11px] text-muted-foreground opacity-60">{tokenLabel}</span>
				{/if}

				{#if level.hasInfo}
					<Tooltip.Root>
						<Tooltip.Trigger>
							<Info class="h-3.5 w-3.5 shrink-0 text-muted-foreground" />
						</Tooltip.Trigger>

						<Tooltip.Content side="right">
							<p>Maximum reasoning effort with extended context usage</p>
						</Tooltip.Content>
					</Tooltip.Root>
				{/if}
			</DropdownMenu.Item>
		{/each}
	</DropdownMenu.Content>
</DropdownMenu.Root>
