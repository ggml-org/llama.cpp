<script lang="ts">
	import { Button } from '$lib/components/ui/button';
	import { APP_NAME } from '$lib/constants';
	import type { PerformanceReading } from '$lib/services/performance.service';
	import { PerformanceService } from '$lib/services/performance.service';
	import { serverStore } from '$lib/stores';
	import { onMount } from 'svelte';

	/** How often the view re-reads the server — rates need repeated samples. */
	const REFRESH_MS = 5000;

	let status = $state<'loading' | 'ok' | 'no-metrics' | 'offline'>('loading');
	let reading = $state<PerformanceReading | null>(null);
	let errorMessage = $state<string | null>(null);

	const props = $derived(serverStore.props);

	function rate(value: number | null): string {
		return value === null ? '—' : `${value.toFixed(1)} tok/s`;
	}

	function bareRate(value: number | null): string {
		return value === null ? '—' : value.toFixed(1);
	}

	function count(value: number | null): string {
		return value === null ? '—' : Math.round(value).toString();
	}

	function percent(value: number | null): string {
		return value === null ? '—' : `${Math.round(value * 100)}%`;
	}

	async function load(): Promise<void> {
		try {
			reading = await PerformanceService.read();
			status = reading.status;
			errorMessage = null;
		} catch (error) {
			// keep the last good reading on refresh failures; only a first-load
			// failure blanks the view into the offline state
			errorMessage = error instanceof Error ? error.message : String(error);

			if (!reading) {
				status = 'offline';
			}
		}
	}

	onMount(() => {
		void load();

		const timer = setInterval(() => void load(), REFRESH_MS);

		return () => clearInterval(timer);
	});
</script>

<svelte:head>
	<title>Performance - {APP_NAME}</title>
</svelte:head>

<div class="mx-auto flex h-[100dvh] w-full max-w-7xl flex-col gap-6 px-4 pt-12 md:px-8 md:pt-6">
	<div class="flex min-w-0 flex-col gap-1">
		<h1 class="text-2xl font-semibold tracking-tight">Performance</h1>

		<p class="text-sm text-muted-foreground">
			Live throughput and slot activity reported by this server.
		</p>
	</div>

	{#if status === 'offline' && !reading}
		<div class="flex flex-col items-center gap-3 py-16 text-center">
			<p class="font-medium">Server unreachable</p>

			<p class="text-sm text-muted-foreground">
				{errorMessage ?? 'The server did not answer.'}
			</p>

			<Button onclick={() => void load()} size="sm" variant="outline">Retry</Button>
		</div>
	{:else if !reading}
		<p class="text-sm text-muted-foreground">Loading performance data…</p>
	{:else}
		{#if errorMessage}
			<p class="text-sm text-destructive">
				{errorMessage} — showing the last successful read.
			</p>
		{/if}

		{#if status === 'no-metrics'}
			<p class="text-sm text-muted-foreground">
				Start the server with <code class="rounded bg-muted px-1 py-0.5">--metrics</code> for server-wide
				throughput; the slot counters below still update live.
			</p>
		{/if}

		{#if props}
			<p class="break-all text-sm text-muted-foreground">
				{props.model_alias ?? props.model_path ?? 'model'} ·
				{props.model_ftype ?? 'unknown type'} ·
				{count(props.total_slots ?? null)} slots · context {count(
					props.default_generation_settings?.n_ctx ?? null
				)}
			</p>
		{/if}

		<div class="flex min-h-0 flex-1 flex-col gap-6 overflow-auto pb-6">
			{#if status === 'ok'}
				<table class="w-full table-fixed border-collapse text-sm">
					<caption class="pb-2 text-left font-medium">Server throughput</caption>

					<tbody class="divide-y divide-border/50">
						<tr>
							<th class="py-2 pr-4 text-left font-normal text-muted-foreground"
								>Prompt processing</th
							>

							<td class="py-2 text-right tabular-nums">{rate(reading.promptPerSec)}</td>
						</tr>

						<tr>
							<th class="py-2 pr-4 text-left font-normal text-muted-foreground">Generation</th>

							<td class="py-2 text-right tabular-nums">{rate(reading.decodePerSec)}</td>
						</tr>

						<tr>
							<th class="py-2 pr-4 text-left font-normal text-muted-foreground">KV-cache hit</th>

							<td class="py-2 text-right tabular-nums">{percent(reading.cacheHitRate)}</td>
						</tr>

						<tr>
							<th class="py-2 pr-4 text-left font-normal text-muted-foreground"
								>Requests processing</th
							>

							<td class="py-2 text-right tabular-nums">{count(reading.requestsProcessing)}</td>
						</tr>

						<tr>
							<th class="py-2 pr-4 text-left font-normal text-muted-foreground"
								>Requests deferred</th
							>

							<td class="py-2 text-right tabular-nums">{count(reading.requestsDeferred)}</td>
						</tr>
					</tbody>
				</table>
			{/if}

			<table class="min-w-[640px] w-full table-fixed border-collapse text-sm">
				<caption class="pb-2 text-left font-medium">Slots</caption>

				<thead>
					<tr class="border-b border-border/50 text-left text-muted-foreground">
						<th class="py-2 pr-2 font-medium">#</th>

						<th class="py-2 pr-2 font-medium">State</th>

						<th class="py-2 pr-2 font-medium">Context</th>

						<th class="py-2 pr-2 font-medium">Prompt</th>

						<th class="py-2 pr-2 font-medium">From cache</th>

						<th class="py-2 pr-2 font-medium">Decoded</th>

						<th class="py-2 pr-2 text-right font-medium">Prompt tok/s</th>

						<th class="py-2 text-right font-medium">Decoded tok/s</th>
					</tr>
				</thead>

				<tbody class="divide-y divide-border/50">
					{#if reading.slots.length === 0}
						<tr>
							<td class="py-3 text-muted-foreground" colspan="8">No slots reported.</td>
						</tr>
					{:else}
						{#each reading.slots as slot (slot.id)}
							<tr>
								<td class="py-2 pr-2 tabular-nums">{slot.id}</td>

								<td class="py-2 pr-2">{slot.isProcessing ? 'busy' : 'idle'}</td>

								<td class="py-2 pr-2 tabular-nums">{count(slot.nCtx)}</td>

								<td class="py-2 pr-2 tabular-nums">{count(slot.promptProcessed)}</td>

								<td class="py-2 pr-2 tabular-nums">{count(slot.promptCached)}</td>

								<td class="py-2 pr-2 tabular-nums">{count(slot.decoded)}</td>

								<td class="py-2 pr-2 text-right tabular-nums">{bareRate(slot.promptPerSec)}</td>

								<td class="py-2 text-right tabular-nums">{bareRate(slot.decodePerSec)}</td>
							</tr>
						{/each}
					{/if}
				</tbody>
			</table>
		</div>
	{/if}
</div>
