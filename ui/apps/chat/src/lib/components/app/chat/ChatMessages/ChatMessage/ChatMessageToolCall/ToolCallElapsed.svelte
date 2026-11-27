<script lang="ts">
	// Time a loop of tool calls took, or since it started while it runs.

	import { useNow } from '#lib/hooks/use-now.svelte.ts';
	import { formatElapsedTime, formatPerformanceTime } from '#lib/utils/index.ts';

	interface Props {
		/** Loop start (ms since epoch). */
		startedAt: number;
		/** Loop end (ms since epoch), or null while it runs. */
		endedAt?: number | null;
	}

	let { endedAt = null, startedAt }: Props = $props();

	const now = useNow(() => endedAt === null);

	const label = $derived(
		endedAt === null
			? formatElapsedTime(now.current - startedAt)
			: formatPerformanceTime(endedAt - startedAt)
	);
</script>

<span
	class="text-body-sm font-normal text-muted-foreground tabular-nums"
	title={endedAt === null ? 'Time since the loop started' : 'Time the loop took'}
>
	{label}
</span>
