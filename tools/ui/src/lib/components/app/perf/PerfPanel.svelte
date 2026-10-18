<script lang="ts">
	import { perfStore } from '$lib/stores/perf.svelte';
	import {
		formatPerfBandwidth,
		formatPerfGb,
		formatPerfMemory,
		formatPerfPercent,
		formatPerfRatio,
		formatPerfTps,
		PERF_NA,
		PERF_PENDING
	} from '$lib/utils/perf-format';

	interface Props {
		/** card: the desktop dashboard; strip: one compact line above the phone composer, tap to expand */
		variant?: 'card' | 'strip';
	}

	let { variant = 'card' }: Props = $props();

	let pinned = $state(false);
	let hovered = $state(false);
	let expanded = $derived(pinned || hovered);
	// the last poll failed: the figures shown are older than they look
	let stale = $derived(perfStore.error !== null && perfStore.data !== null);

	const WASTED_DEFINITION =
		'Preload wasted: bytes read for expert guesses that were evicted before use, or read after their layer had started, as a share of all bytes this server read from disk since the model was loaded.';

	let rows = $derived.by(() => {
		const d = perfStore.data;
		const cpu = d?.cpu;
		const gpu = d?.gpu;
		const mem = d?.mem;
		const gen = d?.gen;
		const prompt = d?.prompt;
		const nvme = d?.nvme;
		const predict = d?.predict;
		const preload = d?.preload;
		const batch =
			prompt?.batch != null && prompt?.n_batches != null
				? `batch ${prompt.batch}/${prompt.n_batches}, `
				: '';
		const layer =
			prompt?.layer != null && prompt?.n_layer != null
				? `layer ${prompt.layer}/${prompt.n_layer}, `
				: '';
		// while a request generates, its speed is known from the second token on
		const genValue =
			gen?.generating && gen.tps_headline == null ? PERF_PENDING : formatPerfTps(gen?.tps_headline);
		const share =
			preload && preload.bytes_read_total > 0
				? formatPerfRatio(preload.bytes_read_predictor / preload.bytes_read_total)
				: PERF_NA;

		return [
			{
				detail: `process ${formatPerfPercent(cpu?.proc_pct)}`,
				label: 'CPU',
				value: formatPerfPercent(cpu?.busy_pct)
			},
			{ detail: gpu?.name ?? PERF_NA, label: 'GPU', value: formatPerfPercent(gpu?.busy_pct) },
			{
				detail: `used ${formatPerfGb(mem?.used_bytes)}, cached ${formatPerfGb(mem?.cached_bytes)}, free ${formatPerfGb(mem?.free_bytes)}`,
				label: 'RAM',
				value: formatPerfMemory(mem?.occupied_bytes, mem?.total_bytes)
			},
			{
				detail: 'device-wide, as NVML counts it (includes memory the driver reserves)',
				label: 'VRAM',
				value: formatPerfMemory(gpu?.mem_used_bytes, gpu?.mem_total_bytes)
			},
			{
				detail: `window ${formatPerfTps(gen?.tps_window)}, lifetime ${formatPerfTps(gen?.tps_total)}`,
				label: 'gen',
				value: genValue
			},
			{
				detail: `${prompt?.estimate ? 'estimated final rate; ' : ''}${batch}${layer}window ${formatPerfTps(prompt?.tps_window)}, lifetime ${formatPerfTps(prompt?.tps_total)}`,
				label: 'pp',
				value: formatPerfTps(prompt?.tps_headline)
			},
			{
				detail: `10 s ${formatPerfBandwidth(nvme?.read_bps_10s)}, ${nvme?.device ?? PERF_NA}`,
				label: 'NVMe',
				value: formatPerfBandwidth(nvme?.read_bps)
			},
			{
				detail: `recall ${formatPerfRatio(predict?.recall_window)}; lifetime: right ${formatPerfRatio(predict?.precision_total)}, recall ${formatPerfRatio(predict?.recall_total)}`,
				label: 'guesses right',
				value: formatPerfRatio(predict?.precision_window)
			},
			{
				detail: preload
					? `still pending ${formatPerfGb(preload.bytes_pending)}, credited ${formatPerfGb(preload.bytes_credited)}, predictor share ${share}`
					: '',
				label: 'preload wasted',
				value: formatPerfPercent(preload?.wasted_pct)
			}
		];
	});
</script>

{#snippet figures(showDetails: boolean)}
	<dl class="grid grid-cols-[auto_auto] gap-x-3 gap-y-0.5">
		{#each rows as row (row.label)}
			<dt class="text-muted-foreground">{row.label}</dt>

			<dd class="text-right font-medium" data-perf={row.label}>{row.value}</dd>

			{#if showDetails && row.detail}
				<dd class="col-span-2 -mt-0.5 mb-0.5 text-[10px] text-muted-foreground">{row.detail}</dd>
			{/if}
		{/each}
	</dl>

	{#if showDetails}
		<p class="mt-1 text-[10px] text-muted-foreground">{WASTED_DEFINITION}</p>
	{/if}
{/snippet}

{#snippet toggle(open: boolean, extra: string)}
	<button
		aria-expanded={open}
		class={['pointer-events-auto text-[10px] text-muted-foreground hover:text-foreground', extra]}
		onclick={() => (pinned = !pinned)}
		onpointerenter={(e) => {
			if (e.pointerType === 'mouse') hovered = true;
		}}
		onpointerleave={(e) => {
			if (e.pointerType === 'mouse') hovered = false;
		}}
		type="button"
	>
		<span class="sr-only">Performance</span>
		{open ? 'less' : 'details'}
	</button>
{/snippet}

{#if perfStore.unavailable}
	<section
		aria-label="Performance"
		class="pointer-events-none rounded-md border bg-background/90 px-2 py-1 text-[11px] text-muted-foreground"
	>
		Performance: unavailable
	</section>
{:else if variant === 'strip'}
	<section
		aria-label="Performance"
		class={[
			'pointer-events-auto flex w-full items-start gap-2 rounded-md border bg-background/90 px-2 py-1 text-xs tabular-nums',
			stale && 'opacity-60'
		]}
	>
		<div class="min-w-0 flex-1">
			{#if pinned}
				{@render figures(true)}
			{:else}
				<div class="flex flex-wrap gap-x-2 gap-y-0.5">
					{#each rows as row (row.label)}
						<span class="whitespace-nowrap">
							<span class="text-muted-foreground">{row.label}</span>

							<span class="font-medium" data-perf={row.label}>{row.value}</span>
						</span>
					{/each}
				</div>
			{/if}

			{#if stale}
				<span class="text-[10px] text-muted-foreground">stale</span>
			{/if}
		</div>

		{@render toggle(pinned, 'shrink-0 self-start')}
	</section>
{:else}
	<section
		aria-label="Performance"
		class={[
			'pointer-events-none rounded-md border bg-background/90 px-2 py-1 text-[11px] tabular-nums shadow-sm backdrop-blur-sm',
			stale && 'opacity-60'
		]}
	>
		{@render figures(expanded)}

		<div class="mt-0.5 flex items-center justify-between">
			<span class="text-[10px] text-muted-foreground">{stale ? 'stale' : ''}</span>

			{@render toggle(expanded, '')}
		</div>
	</section>
{/if}
