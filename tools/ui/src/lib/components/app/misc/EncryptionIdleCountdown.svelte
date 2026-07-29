<script lang="ts">
	import { Timer } from '@lucide/svelte';

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

	$effect(() => {
		if (!hasDeadline) return;

		now = Date.now();
		const interval = setInterval(() => {
			now = Date.now();
		}, 1000);

		return () => clearInterval(interval);
	});

	function formatRemaining(ms: number): string {
		const totalSeconds = Math.ceil(ms / 1000);
		const minutes = Math.floor(totalSeconds / 60);
		const seconds = totalSeconds % 60;
		return `${minutes}:${String(seconds).padStart(2, '0')}`;
	}
</script>

{#if visible && remainingMs !== null}
	<div
		class="fixed top-4 right-4 z-50 flex items-center gap-1.5 rounded-md border border-amber-500/50 bg-amber-500/10 px-2.5 py-1.5 text-xs font-medium text-amber-700 shadow-sm backdrop-blur-sm dark:text-amber-400"
		role="status"
	>
		<Timer class="h-3.5 w-3.5" />
		<span class="tabular-nums">Locking in {formatRemaining(remainingMs)}</span>
	</div>
{/if}
