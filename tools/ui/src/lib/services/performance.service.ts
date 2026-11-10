/**
 * PerformanceService - Live server performance from /metrics and /slots
 *
 * Reads the Prometheus text of `/metrics` (requires the server's `--metrics`
 * flag) and the per-slot task counters of `/slots`, deriving throughput as
 * counter deltas across refreshes — `/slots` carries no timing fields of its
 * own. The KV-cache share comes from the slot counters too: `/metrics` has
 * no cache family to read. When the server runs without `--metrics` the read
 * reports `no-metrics` instead of failing, so the slot table keeps working.
 *
 * Stateless per the service-layer rules: the previous counters sample lives in
 * module scope so deltas survive across calls; no Svelte runes here.
 */

import { base } from '$app/paths';
import { API_METRICS, API_SLOTS } from '$lib/constants';
import type { ApiSlotData } from '$lib/types';
import { ApiError, apiFetch, getAuthHeaders } from '$lib/utils';

/** One slot as the performance view needs it. */
export interface PerformanceSlot {
	id: number;
	nCtx: number;
	isProcessing: boolean;
	promptProcessed: number | null;
	promptCached: number | null;
	decoded: number | null;
	/** Prompt tokens processed per second since the previous refresh (null until known). */
	promptPerSec: number | null;
	/** Decoded tokens per second since the previous refresh (null until known). */
	decodePerSec: number | null;
}

/** Everything the performance panel renders, in one read. */
export interface PerformanceReading {
	/** `no-metrics` = the server answered without `--metrics`; slots still update. */
	status: 'ok' | 'no-metrics';
	/** Server-wide rates: windowed gauge first, counter delta as fallback. */
	promptPerSec: number | null;
	decodePerSec: number | null;
	requestsProcessing: number | null;
	requestsDeferred: number | null;
	/** Share of prompt tokens served from the KV cache since the previous refresh
	 * (summed slot-counter deltas; `processed` is the computed portion, `cached`
	 * the KV-hit portion). */
	cacheHitRate: number | null;
	slots: PerformanceSlot[];
}

/** Metric sample names as emitted (the endpoint prefixes them with `llamacpp:`). */
const METRIC = {
	decodeRate: 'predicted_tokens_seconds',
	decodeTotal: 'tokens_predicted_total',
	promptRate: 'prompt_tokens_seconds',
	promptTotal: 'prompt_tokens_total',
	requestsDeferred: 'requests_deferred',
	requestsProcessing: 'requests_processing'
} as const;

/** Previous counters, kept between reads so rates can be derived. */
let previous: {
	at: number;
	decodeTotal: number | null;
	promptTotal: number | null;
	slots: Map<number, { cached: number | null; decoded: number | null; processed: number | null }>;
} | null = null;

/** Parses Prometheus exposition text into name -> value (labels stripped). */
function parseMetrics(text: string): Map<string, number> {
	const values = new Map<string, number>();

	for (const line of text.split('\n')) {
		if (line.startsWith('#')) continue;

		const match = /^([a-zA-Z_:][\w:]*)(?:\{[^}]*\})?\s+(-?[\d.eE+]+)\s*$/.exec(line);

		if (match) {
			values.set(match[1], Number(match[2]));
		}
	}

	return values;
}

/** Looks a metric up with and without the emitter's `llamacpp:` prefix. */
function pick(values: Map<string, number>, name: string): number | null {
	const found = values.get(`llamacpp:${name}`) ?? values.get(name);

	return typeof found === 'number' && Number.isFinite(found) ? found : null;
}

/** (current - previous) / seconds; null until two comparable samples exist. */
function deltaRate(current: number | null, prior: number | null, seconds: number): number | null {
	if (current === null || prior === null || seconds <= 0 || current < prior) {
		return null;
	}

	return (current - prior) / seconds;
}

export class PerformanceService {
	/**
	 * Reads metrics and slots as one consistent snapshot.
	 *
	 * @returns Derived reading; `status: 'no-metrics'` when the server runs
	 *   without the `--metrics` flag (slots still included)
	 * @throws {ApiError} On HTTP failures (auth, router-mode requirements)
	 * @throws {Error} On network failures — the panel shows its offline state
	 */
	static async read(): Promise<PerformanceReading> {
		const [metricsText, slots] = await Promise.all([
			PerformanceService.fetchMetricsText(),
			apiFetch<ApiSlotData[]>(API_SLOTS.LIST, { authOnly: true })
		]);
		const at = Date.now();
		const seconds = previous ? (at - previous.at) / 1000 : 0;
		const values = metricsText === null ? new Map<string, number>() : parseMetrics(metricsText);
		const promptTotal = pick(values, METRIC.promptTotal);
		const decodeTotal = pick(values, METRIC.decodeTotal);
		// windowed gauges when present, deltas across refreshes as fallback
		const promptPerSec =
			pick(values, METRIC.promptRate) ??
			deltaRate(promptTotal, previous?.promptTotal ?? null, seconds);
		const decodePerSec =
			pick(values, METRIC.decodeRate) ??
			deltaRate(decodeTotal, previous?.decodeTotal ?? null, seconds);
		const slotSnapshots = new Map<
			number,
			{ cached: number | null; decoded: number | null; processed: number | null }
		>();

		let deltaCached = 0;
		let deltaProcessed = 0;

		const performanceSlots: PerformanceSlot[] = slots.map((slot) => {
			const processed = slot.n_prompt_tokens_processed ?? null;
			const cached = slot.n_prompt_tokens_cache ?? null;
			const decoded = slot.next_token?.[0]?.n_decoded ?? null;
			const prior = previous?.slots.get(slot.id) ?? null;

			// positive deltas only: task stats reset when a new task starts
			if (cached !== null && prior !== null && prior.cached !== null && cached >= prior.cached) {
				deltaCached += cached - prior.cached;
			}

			if (
				processed !== null &&
				prior !== null &&
				prior.processed !== null &&
				processed >= prior.processed
			) {
				deltaProcessed += processed - prior.processed;
			}

			slotSnapshots.set(slot.id, { cached, decoded, processed });

			return {
				decoded,
				decodePerSec: deltaRate(decoded, prior?.decoded ?? null, seconds),
				id: slot.id,
				isProcessing: slot.is_processing,
				nCtx: slot.n_ctx,
				promptCached: cached,
				promptPerSec: deltaRate(processed, prior?.processed ?? null, seconds),
				promptProcessed: processed
			};
		});

		previous = { at, decodeTotal, promptTotal, slots: slotSnapshots };
		const cacheHitRate =
			deltaCached + deltaProcessed > 0 ? deltaCached / (deltaCached + deltaProcessed) : null;

		return {
			cacheHitRate,
			decodePerSec,
			promptPerSec,
			requestsDeferred: pick(values, METRIC.requestsDeferred),
			requestsProcessing: pick(values, METRIC.requestsProcessing),
			slots: performanceSlots,
			status: metricsText === null ? 'no-metrics' : 'ok'
		};
	}

	/** Fetches `/metrics` as text; null when the server runs without `--metrics`. */
	private static async fetchMetricsText(): Promise<string | null> {
		const response = await fetch(`${base}${API_METRICS.LIST}`, { headers: getAuthHeaders() });

		if (!response.ok) {
			const body = await response.text();

			// llama-server answers with a JSON error naming the missing flag
			if (body.includes('--metrics')) {
				return null;
			}

			let message = `metrics request failed (${response.status})`;

			try {
				message = (JSON.parse(body) as { error?: string }).error ?? message;
			} catch {
				// non-JSON error body — the status-based message stands
			}

			throw new ApiError(message, response.status);
		}

		const text = await response.text();

		return text.trim().startsWith('{') && text.includes('--metrics') ? null : text;
	}
}
