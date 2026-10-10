<script lang="ts">
	import { Check, Info, Lightbulb, LightbulbOff } from '@lucide/svelte';
	import { Button } from '$lib/components/ui/button';
	import * as Drawer from '$lib/components/ui/drawer';
	import * as DropdownMenu from '$lib/components/ui/dropdown-menu';
	import * as Tooltip from '$lib/components/ui/tooltip';
	import { ICON_CLASS_DEFAULT } from '$lib/constants';
	import { ReasoningEffort } from '$lib/enums';
	import { useReasoningMenu } from '$lib/hooks/use-reasoning-menu.svelte';
	import { deviceStore } from '$lib/stores';
	import type { ReasoningEffortLevel } from '$lib/types';

	const reasoning = useReasoningMenu();

	let isOpen = $state(false);

	// "Default" is the resting state, where the bulb alone carries the meaning
	let isDefault = $derived(reasoning.currentEffort === ReasoningEffort.DEFAULT);
	// a phone picks the level in a drawer, where a thumb reaches it
	let isMobile = $derived(deviceStore.isMobile);

	const MAX_EFFORT_HINT = 'Maximum reasoning effort with extended context usage';

	function select(level: ReasoningEffortLevel): void {
		reasoning.select(level);
		isOpen = false;
	}
</script>

{#snippet triggerLabel()}
	<span class="flex items-center gap-0.75 {reasoning.isOff ? 'text-muted-foreground' : ''}">
		{#if reasoning.isOff}
			<LightbulbOff class="size-3 max-md:size-4 shrink-0" />
		{:else}
			<Lightbulb class="size-3 max-md:size-4 shrink-0" />
		{/if}

		{#if !isDefault}
			<span class="capitalize">{reasoning.currentEffort}</span>
		{/if}
	</span>
{/snippet}

{#snippet levelRow(level: ReasoningEffortLevel)}
	{@const tokenLabel = reasoning.tokenLabel(level)}

	<button
		aria-pressed={reasoning.isSelected(level)}
		class="flex min-h-11 w-full cursor-pointer items-center gap-3 rounded-md px-3 text-left text-sm transition-colors hover:bg-accent"
		onclick={() => select(level)}
		type="button"
	>
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
			<!-- no hover on a phone: the hint rides on the icon's own title there -->
			<Info class="h-3.5 w-3.5 shrink-0 text-muted-foreground" title={MAX_EFFORT_HINT} />
		{/if}
	</button>
{/snippet}

<!-- the level belongs to the chat, not to the model list -->
{#if isMobile}
	<Button
		aria-label="Reasoning effort"
		class="h-auto gap-1 rounded-sm bg-transparent! backdrop-blur-none! {isDefault
			? 'px-1'
			: 'px-1.75!'} py-1 text-xs max-md:h-8"
		onclick={() => (isOpen = true)}
		variant="ghost"
	>
		{@render triggerLabel()}
	</Button>

	<Drawer.Root bind:open={isOpen}>
		<Drawer.Content>
			<Drawer.Header>
				<Drawer.Title>Reasoning effort</Drawer.Title>
			</Drawer.Header>

			<div class="flex flex-col px-2 pb-4">
				{#each reasoning.levels as level (level.value)}
					{@render levelRow(level)}
				{/each}
			</div>
		</Drawer.Content>
	</Drawer.Root>
{:else}
	<DropdownMenu.Root>
		<DropdownMenu.Trigger>
			{#snippet child({ props })}
				<Button
					{...props}
					aria-label="Reasoning effort"
					class="h-auto gap-1 rounded-sm bg-transparent! backdrop-blur-none! {isDefault
						? 'px-1'
						: 'px-1.75!'} py-1 text-xs hover:bg-transparent!"
					variant="ghost"
				>
					{@render triggerLabel()}
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
								<p>{MAX_EFFORT_HINT}</p>
							</Tooltip.Content>
						</Tooltip.Root>
					{/if}
				</DropdownMenu.Item>
			{/each}
		</DropdownMenu.Content>
	</DropdownMenu.Root>
{/if}
