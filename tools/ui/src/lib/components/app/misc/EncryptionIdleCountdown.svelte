<script lang="ts">
	import { TimerReset } from '@lucide/svelte';

	import { encryptionStore } from '$lib/stores/encryption.svelte';

	const WARNING_WINDOW_MS = 60_000;

	let now = $state(Date.now());

	let hasDeadline = $derived(encryptionStore.idleDeadlineAt !== null);

	let remainingMs = $derived(
		encryptionStore.idleDeadlineAt === null ? null : encryptionStore.idleDeadlineAt - now
	);

	let visible = $derived(
		encryptionStore.isUnlocked &&
			remainingMs !== null &&
			remainingMs > 0 &&
			remainingMs <= WARNING_WINDOW_MS
	);

	// How close we are to locking (0 fresh warning .. 1 about to lock); drives the pulse
	let urgency = $derived(
		remainingMs === null ? 0 : Math.min(1, 1 - remainingMs / WARNING_WINDOW_MS)
	);

	$effect(() => {
		if (!hasDeadline) return;

		now = Date.now();
		const interval = setInterval(() => {
			now = Date.now();
		}, 1000);

		return () => clearInterval(interval);
	});

	function formatRemaining(ms: number): string {
		const totalSeconds = Math.max(0, Math.ceil(ms / 1000));
		const minutes = Math.floor(totalSeconds / 60);
		const seconds = totalSeconds % 60;
		return `${minutes}:${String(seconds).padStart(2, '0')}`;
	}
</script>

{#if visible && remainingMs !== null}
	<div
		class="pointer-events-auto fixed top-4 right-4 z-50 relative flex items-center gap-2 rounded-lg border border-amber-500/40 bg-background/90 px-3 pt-2 pb-3 shadow-lg backdrop-blur-md"
		role="status"
		aria-live="polite"
	>
		<span
			class="grid h-6 w-6 place-items-center rounded-full border border-amber-500/40 text-amber-600 dark:text-amber-400 {urgency >=
			0.7
				? 'animate-pulse'
				: ''}"
		>
			<TimerReset class="h-3.5 w-3.5" />
		</span>

		<span class="flex flex-col leading-tight">
			<span class="text-xs font-medium text-foreground">Auto-lock in</span>
			<span class="text-sm font-semibold tabular-nums text-amber-600 dark:text-amber-400">
				{formatRemaining(remainingMs)}
			</span>
		</span>

		<span
			class="absolute right-3 bottom-1.5 left-3 h-0.5 overflow-hidden rounded-full bg-amber-500/15"
		>
			<span
				class="block h-full rounded-full bg-amber-500/60 transition-[width] duration-1000 ease-linear"
				style:width={`{(1 - urgency) * 100}%`}
			></span>
		</span>
	</div>
{/if}
