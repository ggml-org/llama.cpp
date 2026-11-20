<!--
	The chat's reasoning effort, at the foot of the model picker: one row with the level in use, which opens the levels
	under it, sliding open and shut (CollapsibleSection). `menu` is the desktop dropdown's (its rows are menu items),
	`list` the phone drawer's.
-->
<script lang="ts">
	import * as DropdownMenu from '@ggml-org/llama-ui/ui/dropdown-menu';
	import { Check, Lightbulb, LightbulbOff } from '@lucide/svelte';
	import { CollapsibleSection } from '#lib/components/app/index.ts';
	import { ICON_CLASS_DEFAULT, REASONING_EFFORT_LABELS } from '#lib/constants/index.ts';
	import { useReasoningMenu } from '#lib/hooks/use-reasoning-menu.svelte.ts';
	import type { ReasoningEffortLevel } from '#lib/types/index.ts';

	interface Props {
		variant?: 'menu' | 'list';
	}

	let { variant = 'menu' }: Props = $props();

	const reasoning = useReasoningMenu();

	let current = $derived(REASONING_EFFORT_LABELS[reasoning.currentEffort]);
</script>

{#snippet header()}
	{#if reasoning.isOff}
		<LightbulbOff class="size-icon-md shrink-0 text-muted-foreground" />
	{:else}
		<Lightbulb class="size-icon-md shrink-0 text-muted-foreground" />
	{/if}

	<span class="flex-1">Reasoning</span>

	<span class="text-xs text-muted-foreground">{current}</span>
{/snippet}

{#snippet levelLabel(level: ReasoningEffortLevel)}
	{@const tokenLabel = reasoning.tokenLabel(level)}

	{#if reasoning.isSelected(level)}
		<Check class="{ICON_CLASS_DEFAULT} shrink-0 text-foreground" />
	{:else}
		<div class="{ICON_CLASS_DEFAULT} shrink-0"></div>
	{/if}

	<span class="min-w-0 flex-1 truncate">{level.label}</span>

	{#if tokenLabel}
		<span class="shrink-0 text-[11px] text-muted-foreground opacity-60">{tokenLabel}</span>
	{/if}
{/snippet}

{#if variant === 'menu'}
	<DropdownMenu.Group class="px-2" data-slot="models-selector-reasoning">
		<!-- its row is a menu item that opens the levels without closing the menu -->
		<CollapsibleSection defaultOpen={false} inMenu trigger={header}>
			<div class="flex flex-col">
				{#each reasoning.levels as level (level.value)}
					<DropdownMenu.Item class="gap-3 pl-4" onSelect={() => reasoning.select(level)}>
						{@render levelLabel(level)}
					</DropdownMenu.Item>
				{/each}
			</div>
		</CollapsibleSection>
	</DropdownMenu.Group>
{:else}
	<div class="flex flex-col px-2 pt-1" data-slot="models-selector-reasoning">
		<CollapsibleSection
			defaultOpen={false}
			trigger={header}
			triggerClass="flex min-h-11 w-full cursor-pointer items-center gap-2 rounded-md px-3 text-left text-sm transition-colors hover:bg-accent"
		>
			<div class="flex flex-col">
				{#each reasoning.levels as level (level.value)}
					<button
						aria-pressed={reasoning.isSelected(level)}
						class="flex min-h-11 w-full cursor-pointer items-center gap-3 rounded-md pr-3 pl-6 text-left text-sm transition-colors hover:bg-accent"
						onclick={() => reasoning.select(level)}
						type="button"
					>
						{@render levelLabel(level)}
					</button>
				{/each}
			</div>
		</CollapsibleSection>
	</div>
{/if}
