<script lang="ts">
	import { Check, ChevronDown, ChevronUp, Info, Lightbulb, LightbulbOff } from '@lucide/svelte';
	import * as Collapsible from '$lib/components/ui/collapsible';
	import * as DropdownMenu from '$lib/components/ui/dropdown-menu';
	import * as Tooltip from '$lib/components/ui/tooltip';
	import { ICON_CLASS_DEFAULT } from '$lib/constants';
	import { useReasoningMenu } from '$lib/hooks/use-reasoning-menu.svelte';

	const reasoning = useReasoningMenu();

	let expanded = $state(false);
</script>

<!-- Reasoning effort picker for the models dropdown footer; expands in place
     (a flyout submenu would cover the list it belongs to). -->
<Collapsible.Root class="min-w-0" onOpenChange={(open) => (expanded = open)} open={expanded}>
	<!-- A menu item (for keyboard nav) rendering Collapsible.Trigger via `child`;
	     closeOnSelect keeps the menu open while picking. -->
	<DropdownMenu.Item
		class="w-full min-w-0 cursor-pointer items-center gap-2 rounded-md text-left text-sm"
		closeOnSelect={false}
	>
		{#snippet child({ props })}
			<!-- No `class` here: a static attribute would override the spread props.class. -->
			<Collapsible.Trigger {...props}>
				{#if reasoning.isReasoningActive}
					<Lightbulb class="{ICON_CLASS_DEFAULT} shrink-0 text-amber-400" />
				{:else if reasoning.isOff}
					<LightbulbOff class="{ICON_CLASS_DEFAULT} shrink-0 text-muted-foreground" />
				{:else}
					<Lightbulb class="{ICON_CLASS_DEFAULT} shrink-0 text-muted-foreground" />
				{/if}

				<span class="flex min-w-0 flex-1 items-center gap-2">
					<span class="truncate">Reasoning</span>

					<span class="shrink-0 capitalize text-muted-foreground">
						{reasoning.currentEffort}
					</span>
				</span>

				{#if expanded}
					<ChevronUp class="{ICON_CLASS_DEFAULT} shrink-0 text-muted-foreground" />
				{:else}
					<ChevronDown class="{ICON_CLASS_DEFAULT} shrink-0 text-muted-foreground" />
				{/if}
			</Collapsible.Trigger>
		{/snippet}
	</DropdownMenu.Item>

	<Collapsible.Content>
		<!-- Gate on `expanded` so collapsed rows leave the menu item and tab order. -->
		{#if expanded}
			<!-- Plain items (not a RadioGroup): selection lives in the store. -->
			<div class="mt-0.5 flex flex-col gap-0.5 pl-4">
				{#each reasoning.levels as level (level.value)}
					{@const tokenLabel = reasoning.tokenLabel(level)}
					<DropdownMenu.Item
						class="flex w-full cursor-pointer gap-3 px-2 py-1.5"
						closeOnSelect={false}
						onSelect={() => {
							reasoning.select(level);

							// collapse so the footer returns to its resting single-row look
							expanded = false;
						}}
					>
						{#if reasoning.isSelected(level)}
							<Check class="{ICON_CLASS_DEFAULT} shrink-0 text-foreground" />
						{:else}
							<div class="{ICON_CLASS_DEFAULT} shrink-0"></div>
						{/if}

						<span class="min-w-0 flex-1 truncate">{level.label}</span>

						{#if tokenLabel}
							<span class="shrink-0 text-[11px] text-muted-foreground opacity-60">
								{tokenLabel}
							</span>
						{/if}

						{#if level.hasInfo}
							<Tooltip.Root>
								<Tooltip.Trigger>
									<Info class="h-3.5 w-3.5 shrink-0 text-muted-foreground" />
								</Tooltip.Trigger>

								<Tooltip.Content side="left">
									<p>Maximum reasoning effort with extended context usage</p>
								</Tooltip.Content>
							</Tooltip.Root>
						{/if}
					</DropdownMenu.Item>
				{/each}
			</div>
		{/if}
	</Collapsible.Content>
</Collapsible.Root>
