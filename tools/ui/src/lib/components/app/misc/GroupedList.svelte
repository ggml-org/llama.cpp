<script lang="ts" module>
	/** One group of a grouped list, rendered under its own heading. */
	export interface GroupedListGroup<G, E> {
		entries: E[];
		group: G;
		key: string;
	}
</script>

<script generics="G, E" lang="ts">
	import CollapsibleRegion from './CollapsibleRegion.svelte';
	import { modelsStore } from '$lib/stores';
	import type { Snippet } from 'svelte';
	import { SvelteMap, SvelteSet } from 'svelte/reactivity';

	interface Props {
		/** Heading of one group. Omit it for a list whose rows need no heading. */
		group?: Snippet<
			[{ depth: number; expanded: boolean; group: G; key: string; toggle: () => void }]
		>;
		/** Groups to render. Omit to render `items` as a flat list. */
		groups?: GroupedListGroup<G, E>[] | null;
		/** Entries of one group shown before its show more row. 0 shows every entry. */
		groupWindow?: number;
		/** Namespace of the persisted open state, e.g. a section key. Omit to keep it in memory. */
		groupStateKey?: string;
		/** One row. Depth is 1 under a group, 0 in a flat list. */
		item: Snippet<[{ depth: number; entry: E }]>;
		/** Identity of a row, used for the keyed each. */
		keyOf: (entry: E) => string;
		/** The show more row. Without it the rest of a window stays hidden. */
		more?: Snippet<[{ count: number; onMore: () => void; unit: 'entries' | 'families' }]>;
		/** Flat entries, rendered when the list has no groups. */
		items?: E[];
		/** Entries of the list shown before its show more row. 0 shows every entry. */
		sectionWindow?: number;
		/** Sticky class of a group heading, so a surface can shade it differently. */
		stickyClass?: string;
		/** Sticky offset of a group heading, e.g. `top: 2.25rem`. Empty keeps it scrolling. */
		stickyStyle?: string;
		/** Weight of one entry against a window, e.g. the rows it renders. */
		weightOf?: (entry: E) => number;
	}

	let {
		group,
		groups = null,
		groupStateKey,
		groupWindow = 0,
		item,
		items = [],
		keyOf,
		more,
		sectionWindow = 0,
		stickyClass = 'bg-popover',
		stickyStyle = '',
		weightOf
	}: Props = $props();

	// groups start open; this tracks the ones the user folded away, and a list with
	// a state key opens with the ones the user had folded away before the reload
	// the initial namespace only
	// svelte-ignore state_referenced_locally
	const collapsed = new SvelteSet<string>(
		groupStateKey ? modelsStore.collapsedGroupsUnder(groupStateKey) : []
	);
	// windows grow one step at a time, per group and per list
	const groupSteps = new SvelteMap<string, number>();
	let sectionSteps = $state(0);

	function toggleGroup(key: string): void {
		if (collapsed.has(key)) {
			collapsed.delete(key);
		} else {
			collapsed.add(key);
		}

		if (groupStateKey) modelsStore.setGroupOpen(`${groupStateKey}-${key}`, !collapsed.has(key));
	}

	function growGroup(key: string): void {
		groupSteps.set(key, (groupSteps.get(key) ?? 0) + 1);
	}

	let groupCap = $derived((key: string) =>
		groupWindow > 0 ? groupWindow * (1 + (groupSteps.get(key) ?? 0)) : Infinity
	);
	let weight = $derived(weightOf ?? (() => 1));

	/** Groups with their entries cut to the window, and what each cut leaves behind. */
	let windowed = $derived.by((): Array<{ group: G; hidden: number; key: string; rows: E[] }> => {
		const shown: Array<{ group: G; hidden: number; key: string; rows: E[] }> = [];
		const cap = sectionWindow > 0 ? sectionWindow + sectionSteps : Infinity;

		let used = 0;

		for (const entry of groups ?? []) {
			if (used >= cap) break;

			const rows = entry.entries.slice(0, groupCap(entry.key));

			shown.push({
				group: entry.group,
				hidden: entry.entries.length - rows.length,
				key: entry.key,
				rows
			});
			used += rows.reduce((sum, row) => sum + weight(row), 0);
		}

		return shown;
	});

	let flatRows = $derived(sectionWindow > 0 ? items.slice(0, sectionWindow + sectionSteps) : items);
</script>

{#if groups}
	{#each windowed as entry (entry.key)}
		{@const expanded = !collapsed.has(entry.key)}

		{#if group}
			<div class="sticky z-10 {stickyClass}" style={stickyStyle}>
				{@render group({
					depth: 0,
					expanded,
					group: entry.group,
					key: entry.key,
					toggle: () => toggleGroup(entry.key)
				})}
			</div>
		{/if}

		<CollapsibleRegion open={expanded}>
			{#each entry.rows as row (keyOf(row))}
				{@render item({ depth: 1, entry: row })}
			{/each}

			{#if entry.hidden > 0 && more}
				{@render more({
					count: Math.min(groupWindow, entry.hidden),
					onMore: () => growGroup(entry.key),
					unit: 'entries'
				})}
			{/if}
		</CollapsibleRegion>
	{/each}

	{#if windowed.length < (groups?.length ?? 0) && more}
		{@render more({
			count: Math.min(sectionWindow, (groups?.length ?? 0) - windowed.length),
			onMore: () => (sectionSteps += 1),
			unit: 'families'
		})}
	{/if}
{:else}
	{#each flatRows as row (keyOf(row))}
		{@render item({ depth: 0, entry: row })}
	{/each}

	{#if flatRows.length < items.length && more}
		{@render more({
			count: Math.min(sectionWindow, items.length - flatRows.length),
			onMore: () => (sectionSteps += 1),
			unit: 'entries'
		})}
	{/if}
{/if}
