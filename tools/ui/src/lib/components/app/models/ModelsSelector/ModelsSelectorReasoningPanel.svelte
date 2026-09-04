<script lang="ts">
	import { Check, ChevronDown, ChevronUp, Info, Lightbulb, LightbulbOff } from '@lucide/svelte';
	import * as DropdownMenu from '$lib/components/ui/dropdown-menu';
	import * as Tooltip from '$lib/components/ui/tooltip';
	import { ICON_CLASS_DEFAULT } from '$lib/constants';
	import { useReasoningMenu } from '$lib/hooks/use-reasoning-menu.svelte';
	import type { ReasoningEffortLevel } from '$lib/types';

	interface Props {
		/** Render rows as DropdownMenu items (desktop dropdown); plain buttons in menus without one (sheet). */
		inMenu?: boolean;
	}

	let { inMenu = false }: Props = $props();

	const reasoning = useReasoningMenu();

	let expanded = $state(false);

	// rows stay mounted through the collapse transition so it can play; after
	// it, menus drop them - mounted rows are menu items and would pollute the
	// arrow-key navigation
	// measured px transition: two plain lengths interpolate in every browser,
	// no calc-size() support needed. The rows mount at height 0, get measured,
	// then the region grows to the measured height.
	const EXPAND_TRANSITION_MS = 200;
	let rowsMounted = $state(false);
	let regionEl = $state<HTMLDivElement | null>(null);
	let heightPx = $state(0);

	$effect(() => {
		if (!inMenu) {
			rowsMounted = true;
		}

		if (expanded) {
			// in menus the rows mount with the flip; measure once they are laid out
			if (inMenu) rowsMounted = true;

			requestAnimationFrame(() => {
				if (regionEl) heightPx = regionEl.scrollHeight;
			});
		} else {
			heightPx = 0;

			// menus drop the rows after the collapse: mounted rows are menu
			// items and would pollute the arrow-key navigation
			if (inMenu) {
				const timer = setTimeout(() => (rowsMounted = false), EXPAND_TRANSITION_MS);

				return () => clearTimeout(timer);
			}
		}
	});

	const pickLevel = (level: ReasoningEffortLevel) => {
		reasoning.select(level);

		// collapse so the footer returns to its resting single-row look
		expanded = false;
	};
</script>

<!-- Reasoning effort picker for the models selector footer; expands in place
     (a flyout submenu would cover the list it belongs to). Rows are rendered
     as dropdown menu items inside the dropdown, plain buttons in the sheet,
     which has no menu context. -->
{#snippet triggerContent()}
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
{/snippet}

{#if inMenu}
	<!-- A menu item (for keyboard nav) wrapping the trigger via `child`;
	     closeOnSelect keeps the menu open while picking. -->
	<DropdownMenu.Item
		class="w-full min-w-0 cursor-pointer items-center gap-2 rounded-md text-left text-sm"
		closeOnSelect={false}
	>
		{#snippet child({ props })}
			<!-- No `class` here: a static attribute would override the spread props.class. -->
			<button
				{...props}
				aria-expanded={expanded}
				onclick={() => (expanded = !expanded)}
				type="button"
			>
				{@render triggerContent()}
			</button>
		{/snippet}
	</DropdownMenu.Item>
{:else}
	<button
		aria-expanded={expanded}
		class="flex w-full min-w-0 cursor-pointer items-center gap-2 rounded-md px-2 py-1.5 text-left text-sm outline-none transition-colors hover:bg-accent focus-visible:bg-accent"
		onclick={() => (expanded = !expanded)}
		type="button"
	>
		{@render triggerContent()}
	</button>
{/if}

<!-- Custom expand region instead of bits-ui Collapsible (whose conditional
     rendering kills the transition): the height animates through
     calc-size(auto, size) and visibility keeps collapsed rows out of the tab
     order. In menus the rows unmount after the collapse transition, so the
     arrow-key navigation never sees hidden items. -->
<div
	bind:this={regionEl}
	data-expanded={expanded}
	class="overflow-hidden"
	style={`height: ${heightPx}px; visibility: ${
		expanded ? 'visible' : 'hidden'
	}; transition: height ${EXPAND_TRANSITION_MS}ms cubic-bezier(0.23, 1, 0.32, 1), visibility ${EXPAND_TRANSITION_MS}ms;`}
>
	{#if rowsMounted}
		<!-- Plain items (not a RadioGroup): selection lives in the store. -->
		<div class="mt-0.5 flex flex-col gap-0.5 pl-4">
			{#each reasoning.levels as level (level.value)}
				{@const tokenLabel = reasoning.tokenLabel(level)}
				{#if inMenu}
					<DropdownMenu.Item
						class="flex w-full cursor-pointer gap-3 px-2 py-1.5"
						closeOnSelect={false}
						onSelect={() => pickLevel(level)}
					>
						{@render levelContent(level, tokenLabel)}
					</DropdownMenu.Item>
				{:else}
					<button
						class="flex w-full cursor-pointer items-center gap-3 rounded-md px-2 py-1.5 text-left text-sm outline-none transition-colors hover:bg-accent focus-visible:bg-accent"
						onclick={() => pickLevel(level)}
						type="button"
					>
						{@render levelContent(level, tokenLabel)}
					</button>
				{/if}
			{/each}
		</div>
	{/if}
</div>

{#snippet levelContent(level: ReasoningEffortLevel, tokenLabel: string | null)}
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
{/snippet}
