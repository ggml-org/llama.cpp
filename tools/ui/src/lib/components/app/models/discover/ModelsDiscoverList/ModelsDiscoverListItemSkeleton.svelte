<script lang="ts">
	import { Skeleton } from '$lib/components/ui/skeleton';

	interface Props {
		/** Row index; varies skeleton widths so the list does not look mechanical. */
		index?: number;
	}

	let { index = 0 }: Props = $props();

	// Deterministic width variations keyed by row position.
	const NAME_WIDTHS = ['w-40', 'w-52', 'w-44', 'w-56'];
	const BADGE_WIDTHS = [
		['w-12', 'w-14', 'w-10'],
		['w-16', 'w-12', 'w-12'],
		['w-10', 'w-16', 'w-10'],
		['w-14', 'w-10', 'w-14']
	];

	let nameWidth = $derived(NAME_WIDTHS[index % NAME_WIDTHS.length]);
	let badgeWidths = $derived(BADGE_WIDTHS[index % BADGE_WIDTHS.length]);
</script>

<!-- Static skeleton of ModelsDiscoverListItem: avatar, name and badge rows. -->
<li>
	<div class="flex w-full items-start gap-2.5 rounded-lg p-2.5 text-left">
		<Skeleton class="h-9 w-9 shrink-0 rounded-md" />

		<div class="min-w-0 flex-1 space-y-1.5">
			<Skeleton class="{nameWidth} h-4 max-w-full" />

			<div class="flex items-center gap-1">
				{#each badgeWidths as width, i (i)}
					<Skeleton class="{width} h-3.5 rounded" />
				{/each}
			</div>
		</div>
	</div>
</li>
