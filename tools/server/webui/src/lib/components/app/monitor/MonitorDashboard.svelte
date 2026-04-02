<script lang="ts">
	import { onMount } from 'svelte';
	import { base } from '$app/paths';
	import { Activity, CirclePause, CirclePlay, Gauge, Hash, Timer, Zap } from '@lucide/svelte';
	import * as Card from '$lib/components/ui/card';
	import { Badge } from '$lib/components/ui/badge';
	import { getAuthHeaders } from '$lib/utils/api-headers';

	interface MonitorSlot {
		id: number;
		is_processing: boolean;
		n_ctx: number;
		id_task?: number;
		n_decoded?: number;
		n_remain?: number;
		has_next_token?: boolean;
		prompt?: string;
		generated?: string;
	}

	interface MonitorMetrics {
		prompt_tokens_per_second: number;
		predicted_tokens_per_second: number;
		prompt_tokens_total: number;
		predicted_tokens_total: number;
		n_decode_total: number;
	}

	interface MonitorEvent {
		timestamp_ms: number;
		uptime_seconds: number;
		idle_slots: number;
		processing_slots: number;
		slots: MonitorSlot[];
		metrics: MonitorMetrics;
	}

	interface SlotHistory {
		prompt: string;
		generated: string;
		n_decoded: number;
		finished_at: number;
	}

	let data: MonitorEvent | null = $state(null);
	let connected = $state(false);
	let abortController: AbortController | null = null;
	let reconnectTimeout: ReturnType<typeof setTimeout> | null = null;
	let reconnectDelay = 1000;
	let fatalError: string | null = $state(null);
	let uptimeOffset = $state(0); // seconds elapsed since last SSE event
	let uptimeInterval: ReturnType<typeof setInterval> | null = null;

	// track last activity per slot so we can show it after slot goes idle
	let slotHistory: Map<number, SlotHistory> = $state(new Map());
	// track which slots were processing in the previous event
	let prevProcessing: Set<number> = new Set();

	function formatUptime(seconds: number): string {
		if (seconds < 0) return '—';
		const d = Math.floor(seconds / 86400);
		const h = Math.floor((seconds % 86400) / 3600);
		const m = Math.floor((seconds % 3600) / 60);
		if (d > 0) return `${d}d ${h}h ${m}m`;
		if (h > 0) return `${h}h ${m}m`;
		return `${m}m`;
	}

	function truncate(text: string, maxLen: number): string {
		if (!text) return '—';
		// strip special tokens commonly added by various model families
		let clean = text
			.replace(/^(<s>|<\|begin_of_text\|>|<\|im_start\|>system\n?|<\|startoftext\|>|\[BOS\])\s*/i, '')
			.replace(/(<\/s>|<\|end_of_text\|>|<\|im_end\|>|<\|endoftext\|>|\[EOS\])\s*$/i, '')
			.trim();
		if (!clean) return '—';
		if (clean.length <= maxLen) return clean;
		return clean.slice(0, maxLen) + '…';
	}

	function formatTime(ms: number): string {
		if (!ms) return '';
		const d = new Date(ms);
		return d.toLocaleTimeString();
	}

	function timeAgo(ms: number, nowMs: number): string {
		const sec = Math.floor((nowMs - ms) / 1000);
		if (sec < 5) return 'just now';
		if (sec < 60) return `${sec}s ago`;
		const min = Math.floor(sec / 60);
		return `${min}m ${sec % 60}s ago`;
	}

	function processEvent(parsed: MonitorEvent) {
		for (const slot of parsed.slots) {
			if (slot.is_processing) {
				slotHistory.set(slot.id, {
					prompt: slot.prompt || '',
					generated: slot.generated || '',
					n_decoded: slot.n_decoded || 0,
					finished_at: 0,
				});
			} else if (prevProcessing.has(slot.id)) {
				const prev = slotHistory.get(slot.id);
				if (prev) {
					prev.finished_at = parsed.timestamp_ms;
					slotHistory.set(slot.id, prev);
				}
			} else if (!slotHistory.has(slot.id) && (slot.prompt || slot.generated)) {
				slotHistory.set(slot.id, {
					prompt: slot.prompt || '',
					generated: slot.generated || '',
					n_decoded: slot.n_decoded || 0,
					finished_at: parsed.timestamp_ms,
				});
			}
		}

		prevProcessing = new Set(parsed.slots.filter((s) => s.is_processing).map((s) => s.id));
		uptimeOffset = 0;
		data = parsed;
	}

	async function connect() {
		disconnect();
		fatalError = null;

		abortController = new AbortController();

		let response: Response;
		try {
			response = await fetch(`${base}/monitor`, {
				headers: getAuthHeaders(),
				signal: abortController.signal,
			});
		} catch {
			scheduleReconnect();
			return;
		}

		// fatal errors - do not reconnect
		if (response.status === 401 || response.status === 403) {
			fatalError = 'Authentication required. Set API key in settings.';
			return;
		}
		if (response.status === 501) {
			fatalError = 'Monitor not available. Server needs --slots flag.';
			return;
		}
		if (response.status === 503) {
			fatalError = 'Too many monitor clients connected. Try again later.';
			return;
		}
		if (!response.ok || !response.body) {
			scheduleReconnect();
			return;
		}

		connected = true;
		reconnectDelay = 1000;
		uptimeInterval = setInterval(() => { uptimeOffset++; }, 1000);

		const reader = response.body.getReader();
		const decoder = new TextDecoder();
		let buffer = '';

		try {
			while (true) {
				const { done, value } = await reader.read();
				if (done) {
					buffer += decoder.decode(); // flush remaining bytes
					break;
				}

				buffer += decoder.decode(value, { stream: true });

				// parse SSE events from buffer
				const parts = buffer.split('\n\n');
				buffer = parts.pop() || '';

				for (const part of parts) {
					const dataLine = part.split('\n').find((l) => l.startsWith('data: '));
					if (dataLine) {
						try {
							processEvent(JSON.parse(dataLine.slice(6)));
						} catch {
							// ignore parse errors
						}
					}
				}
			}
		} catch (err: unknown) {
			if (err instanceof DOMException && err.name === 'AbortError') {
				return; // clean disconnect
			}
		}

		connected = false;
		scheduleReconnect();
	}

	function scheduleReconnect() {
		connected = false;
		reconnectTimeout = setTimeout(connect, reconnectDelay);
		reconnectDelay = Math.min(reconnectDelay * 2, 30000);
	}

	function disconnect() {
		if (uptimeInterval) {
			clearInterval(uptimeInterval);
			uptimeInterval = null;
		}
		if (reconnectTimeout) {
			clearTimeout(reconnectTimeout);
			reconnectTimeout = null;
		}
		if (abortController) {
			abortController.abort();
			abortController = null;
		}
		connected = false;
	}

	onMount(() => {
		connect();
		return disconnect;
	});
</script>

<div class="flex h-full flex-col overflow-auto p-4 md:p-6">
	<!-- Header -->
	<div class="mb-6 flex items-center justify-between">
		<div class="flex items-center gap-3">
			<Activity class="h-6 w-6 text-foreground" />
			<h1 class="text-2xl font-semibold text-foreground">Server Monitor</h1>
		</div>

		<Badge variant={connected ? 'default' : 'destructive'}>
			<span
				class="mr-1 inline-block h-2 w-2 rounded-full {connected
					? 'bg-green-400 animate-pulse'
					: 'bg-red-400'}"
			></span>
			{connected ? 'Connected' : 'Disconnected'}
		</Badge>
	</div>

	{#if data}
		<!-- Overview cards -->
		<div class="mb-6 grid grid-cols-2 gap-3 md:grid-cols-4">
			<Card.Root>
				<Card.Header class="pb-2">
					<Card.Description class="flex items-center gap-1.5 text-xs">
						<Hash class="h-3.5 w-3.5" />
						Total Slots
					</Card.Description>
				</Card.Header>
				<Card.Content>
					<p class="text-2xl font-bold">{data.slots.length}</p>
				</Card.Content>
			</Card.Root>

			<Card.Root>
				<Card.Header class="pb-2">
					<Card.Description class="flex items-center gap-1.5 text-xs">
						<CirclePlay class="h-3.5 w-3.5 text-yellow-500" />
						Processing
					</Card.Description>
				</Card.Header>
				<Card.Content>
					<p class="text-2xl font-bold">{data.processing_slots}</p>
				</Card.Content>
			</Card.Root>

			<Card.Root>
				<Card.Header class="pb-2">
					<Card.Description class="flex items-center gap-1.5 text-xs">
						<CirclePause class="h-3.5 w-3.5 text-green-500" />
						Idle
					</Card.Description>
				</Card.Header>
				<Card.Content>
					<p class="text-2xl font-bold">{data.idle_slots}</p>
				</Card.Content>
			</Card.Root>

			<Card.Root>
				<Card.Header class="pb-2">
					<Card.Description class="flex items-center gap-1.5 text-xs">
						<Timer class="h-3.5 w-3.5" />
						Uptime
					</Card.Description>
				</Card.Header>
				<Card.Content>
					<p class="text-2xl font-bold">
						{formatUptime((data.uptime_seconds ?? 0) + uptimeOffset)}
					</p>
				</Card.Content>
			</Card.Root>
		</div>

		<!-- Aggregate metrics -->
		<div class="mb-6 grid grid-cols-2 gap-3 md:grid-cols-4">
			<Card.Root>
				<Card.Header class="pb-2">
					<Card.Description class="flex items-center gap-1.5 text-xs">
						<Zap class="h-3.5 w-3.5" />
						Prompt Speed
					</Card.Description>
				</Card.Header>
				<Card.Content>
					<p class="text-xl font-bold">{(data.metrics?.prompt_tokens_per_second ?? 0).toFixed(1)}</p>
					<p class="text-xs text-muted-foreground">tok/s</p>
				</Card.Content>
			</Card.Root>

			<Card.Root>
				<Card.Header class="pb-2">
					<Card.Description class="flex items-center gap-1.5 text-xs">
						<Gauge class="h-3.5 w-3.5" />
						Generation Speed
					</Card.Description>
				</Card.Header>
				<Card.Content>
					<p class="text-xl font-bold">{(data.metrics?.predicted_tokens_per_second ?? 0).toFixed(1)}</p>
					<p class="text-xs text-muted-foreground">tok/s</p>
				</Card.Content>
			</Card.Root>

			<Card.Root>
				<Card.Header class="pb-2">
					<Card.Description class="text-xs">Prompt Tokens Total</Card.Description>
				</Card.Header>
				<Card.Content>
					<p class="text-xl font-bold">{(data.metrics?.prompt_tokens_total ?? 0).toLocaleString()}</p>
				</Card.Content>
			</Card.Root>

			<Card.Root>
				<Card.Header class="pb-2">
					<Card.Description class="text-xs">Generated Tokens Total</Card.Description>
				</Card.Header>
				<Card.Content>
					<p class="text-xl font-bold">
						{(data.metrics?.predicted_tokens_total ?? 0).toLocaleString()}
					</p>
				</Card.Content>
			</Card.Root>
		</div>

		<!-- Slot cards -->
		<h2 class="mb-3 text-lg font-semibold text-foreground">Slots</h2>
		<div class="grid gap-3 md:grid-cols-2">
			{#each data.slots as slot (slot.id)}
				<Card.Root
					class="transition-colors {slot.is_processing ? 'border-yellow-500/40' : 'border-border'}"
				>
					<Card.Header class="pb-2">
						<div class="flex items-center justify-between">
							<Card.Title class="text-sm font-medium">Slot {slot.id}</Card.Title>
							<Badge variant={slot.is_processing ? 'secondary' : 'outline'}>
								{#if slot.is_processing}
									<span class="mr-1 inline-block h-2 w-2 animate-pulse rounded-full bg-yellow-400"
									></span>
									Processing
								{:else}
									<span class="mr-1 inline-block h-2 w-2 rounded-full bg-green-400"></span>
									Idle
								{/if}
							</Badge>
						</div>
					</Card.Header>
					<Card.Content class="space-y-3">
						{#if slot.is_processing}
							<div>
								<p class="mb-1 text-xs font-medium text-muted-foreground">Prompt</p>
								<p
									class="rounded-md bg-muted p-2 font-mono text-xs leading-relaxed text-foreground"
								>
									{truncate(slot.prompt, 200)}
								</p>
							</div>

							<div>
								<p class="mb-1 text-xs font-medium text-muted-foreground">Generated</p>
								<p
									class="max-h-32 overflow-auto rounded-md bg-muted p-2 font-mono text-xs leading-relaxed text-foreground"
								>
									{slot.generated || '…'}
								</p>
							</div>

							<div class="flex flex-wrap gap-3 text-xs text-muted-foreground">
								<span>Decoded: <strong class="text-foreground">{slot.n_decoded ?? 0}</strong></span>
								<span>Remaining: <strong class="text-foreground">{slot.n_remain === -1 ? '∞' : slot.n_remain ?? 0}</strong></span>
								<span>Context: <strong class="text-foreground">{(slot.n_ctx ?? 0).toLocaleString()}</strong></span>
							</div>
						{:else}
							{@const history = slotHistory.get(slot.id)}
							{#if history}
								<div class="opacity-60">
									<div class="mb-2 flex items-center gap-2 text-xs text-muted-foreground">
										<span>Last activity</span>
										{#if history.finished_at}
											<span>· {timeAgo(history.finished_at, data.timestamp_ms)}</span>
											<span>· {formatTime(history.finished_at)}</span>
										{/if}
										<span>· {history.n_decoded} tokens</span>
									</div>
									{#if history.prompt}
										<div class="mb-2">
											<p class="mb-1 text-xs font-medium text-muted-foreground">Prompt</p>
											<p class="rounded-md bg-muted p-2 font-mono text-xs leading-relaxed text-foreground">
												{truncate(history.prompt, 200)}
											</p>
										</div>
									{/if}
									{#if history.generated}
										<div>
											<p class="mb-1 text-xs font-medium text-muted-foreground">Generated</p>
											<p class="max-h-32 overflow-auto rounded-md bg-muted p-2 font-mono text-xs leading-relaxed text-foreground">
												{history.generated}
											</p>
										</div>
									{/if}
									<div class="flex flex-wrap gap-3 text-xs text-muted-foreground">
										<span>Decoded: <strong class="text-foreground">{history.n_decoded}</strong></span>
										<span>Context: <strong class="text-foreground">{(slot.n_ctx ?? 0).toLocaleString()}</strong></span>
									</div>
								</div>
							{:else}
								<p class="text-sm text-muted-foreground">Waiting for request...</p>
							{/if}
						{/if}
					</Card.Content>
				</Card.Root>
			{/each}
		</div>
	{:else if fatalError}
		<div class="flex flex-1 items-center justify-center">
			<p class="text-destructive">{fatalError}</p>
		</div>
	{:else}
		<div class="flex flex-1 items-center justify-center">
			<p class="text-muted-foreground">
				{connected ? 'Waiting for data...' : 'Connecting to server...'}
			</p>
		</div>
	{/if}
</div>
