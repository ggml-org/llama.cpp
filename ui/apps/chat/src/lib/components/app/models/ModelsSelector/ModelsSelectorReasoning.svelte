<!--
	The chat's reasoning effort, at the foot of the model picker: one row with the level in use, which opens the levels
	under it, sliding open and shut (CollapsibleSection). `menu` is the desktop dropdown's (its rows are menu items),
	`list` the phone sheet's, an inset group that opens in place.
-->
<script lang="ts">
	import { InsetList, InsetListRow } from '@ggml-org/llama-ui/app/sheets';
	import * as DropdownMenu from '@ggml-org/llama-ui/ui/dropdown-menu';
	import { Check, Lightbulb, LightbulbOff } from '@lucide/svelte';
	import { CollapsibleSection } from '#lib/components/app/index.ts';
	import {
		ICON_CLASS_DEFAULT,
		REASONING_EFFORT_LABELS,
		REASONING_EFFORT_ROW_LABEL
	} from '#lib/constants/index.ts';
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

	<span class="flex-1">{REASONING_EFFORT_ROW_LABEL}</span>

	<span class="text-caption text-muted-foreground">{current}</span>
{/snippet}

<!-- the phone row, laid out as an InsetListRow: the icon outside the body, the value at its end -->
{#snippet listHeader()}
	{#if reasoning.isOff}
		<LightbulbOff class="{ICON_CLASS_DEFAULT} shrink-0 text-muted-foreground" />
	{:else}
		<Lightbulb class="{ICON_CLASS_DEFAULT} shrink-0 text-muted-foreground" />
	{/if}

	<span class="inset-list-row-body">
		<span class="flex-1">{REASONING_EFFORT_ROW_LABEL}</span>

		<span class="text-muted-foreground">{current}</span>
	</span>
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
		<span class="shrink-0 text-caption text-muted-foreground opacity-60">{tokenLabel}</span>
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
	<!-- an inset group, as the sheet's other rows: the row with the level in use, and the levels
	     sliding open under it inside the same group -->
	<InsetList data-slot="models-selector-reasoning">
		<CollapsibleSection defaultOpen={false} trigger={listHeader} triggerClass="inset-list-row">
			{#each reasoning.levels as level (level.value)}
				{@const tokenLabel = reasoning.tokenLabel(level)}

				<!-- every level keeps its divider, the first one included: the trigger is above it -->
				<InsetListRow
					aria-pressed={reasoning.isSelected(level)}
					bodyClass="border-t!"
					onclick={() => reasoning.select(level)}
				>
					{#snippet leading()}
						{#if reasoning.isSelected(level)}
							<Check class="{ICON_CLASS_DEFAULT} shrink-0 text-(--action-primary)" />
						{:else}
							<span class="{ICON_CLASS_DEFAULT} shrink-0"></span>
						{/if}
					{/snippet}

					<span class="min-w-0 flex-1 truncate">{level.label}</span>

					{#if tokenLabel}
						<span class="shrink-0 text-caption text-muted-foreground">{tokenLabel}</span>
					{/if}
				</InsetListRow>
			{/each}
		</CollapsibleSection>
	</InsetList>
{/if}
