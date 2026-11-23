<script lang="ts">
	// Time since the task started, ticking once a second.

	import { MS_PER_SECOND, SECONDS_PER_MINUTE } from '#lib/constants/index.ts';
	import { formatPerformanceTime } from '#lib/utils/index.ts';

	interface Props {
		/** Task start (ms since epoch). */
		startedAt: number;
	}

	let { startedAt }: Props = $props();

	let now = $state(Date.now());

	$effect(() => {
		const timer = setInterval(() => (now = Date.now()), MS_PER_SECOND);

		return () => clearInterval(timer);
	});

	const elapsedMs = $derived(Math.max(0, now - startedAt));
	// whole seconds while short, so the label does not flicker every tick
	const label = $derived(
		elapsedMs < SECONDS_PER_MINUTE * MS_PER_SECOND
			? `${Math.floor(elapsedMs / MS_PER_SECOND)}s`
			: formatPerformanceTime(elapsedMs)
	);
</script>

<span
	class="text-caption font-normal text-muted-foreground tabular-nums"
	title="Time since the task started"
>
	{label}
</span>
