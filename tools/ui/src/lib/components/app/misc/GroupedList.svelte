<script lang="ts" module>
	/** One group of a grouped list, rendered under its own heading. */
	export interface GroupedListGroup<G, E> {
		entries: E[];
		group: G;
		key: string;
	}

	/** What a show-more row counts: rows of one group, or whole groups of the list. */
	export const GroupedListUnit = {
		ENTRIES: 'entries',
		GROUPS: 'families'
	} as const;

	export type GroupedListUnit = (typeof GroupedListUnit)[keyof typeof GroupedListUnit];
</script>

<script generics="G, E" lang="ts">
	import CollapsibleRegion from './CollapsibleRegion.svelte';
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
		/** Group ids folded away, read once so a caller can restore what it persisted. */
		initialCollapsed?: string[];
		/** One row. Depth is 1 under a group, 0 in a flat list. */
		item: Snippet<[{ depth: number; entry: E }]>;
		/** Identity of a row, used for the keyed each. */
		keyOf: (entry: E) => string;
		/** The show more row. Without it the rest of a window stays hidden. */
		more?: Snippet<[{ count: number; onMore: () => void; unit: GroupedListUnit }]>;
		/** Flat entries, rendered when the list has no groups. */
		items?: E[];
		/** Entries of the list shown before its show more row. 0 shows every entry. */
		sectionWindow?: number;
		/** Called on every group toggle, for a caller that persists the state. */
		onCollapsedChange?: ((key: string, collapsed: boolean) => void) | null;
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
		groupWindow = 0,
		initialCollapsed = [],
		item,
		items = [],
		keyOf,
		more,
		onCollapsedChange = null,
		sectionWindow = 0,
		stickyClass = 'bg-popover',
		stickyStyle = '',
		weightOf
	}: Props = $props();

	// groups start open: this tracks the ones the user folded away, seeded once from
	// whatever the caller persisted
	// svelte-ignore state_referenced_locally
	const collapsed = new SvelteSet<string>(initialCollapsed);
	// windows grow one step at a time, per group and per list
	const groupSteps = new SvelteMap<string, number>();
	let sectionSteps = $state(0);

	function toggleGroup(key: string): void {
		if (collapsed.has(key)) {
			collapsed.delete(key);
		} else {
			collapsed.add(key);
		}

		onCollapsedChange?.(key, collapsed.has(key));
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

	/** Rows one group weighs against the window. */
	const groupWeight = (entry: GroupedListGroup<G, E>) =>
		entry.entries.reduce((sum, row) => sum + weight(row), 0);

	/** The groups one page reveals, and the rows they weigh against the window. */
	let groupPage = $derived.by(() => {
		const page = (groups ?? []).slice(windowed.length, windowed.length + sectionWindow);

		return {
			count: page.length,
			weight: page.reduce((sum, entry) => sum + groupWeight(entry), 0)
		};
	});
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
					unit: GroupedListUnit.ENTRIES
				})}
			{/if}
		</CollapsibleRegion>
	{/each}

	{#if windowed.length < (groups?.length ?? 0) && more}
		{@render more({
			count: groupPage.count,
			onMore: () => (sectionSteps += Math.max(1, groupPage.weight)),
			unit: GroupedListUnit.GROUPS
		})}
	{/if}
{:else}
	{#each flatRows as row (keyOf(row))}
		{@render item({ depth: 0, entry: row })}
	{/each}

	{#if flatRows.length < items.length && more}
		{@render more({
			count: Math.min(sectionWindow, items.length - flatRows.length),
			onMore: () => (sectionSteps += Math.min(sectionWindow, items.length - flatRows.length)),
			unit: GroupedListUnit.ENTRIES
		})}
	{/if}
{/if}
