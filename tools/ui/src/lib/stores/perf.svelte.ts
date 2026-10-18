/**
 * perfStore - Live server performance for the performance panel
 *
 * Owns the polling of GET /perf: one request right away when enabled, then one a second while enabled and the
 * page is visible, none while hidden or disabled. After 3 failures in a row (e.g. 503 while the model loads) it
 * slows down to one request every 5 s; a 404 (a server without /perf) stops it with `unavailable`.
 */

import { browser } from '$app/environment';
import { PerfService } from '$lib/services/perf.service';
import type { ApiPerfResponse } from '$lib/types';
import { ApiError } from '$lib/utils/api-fetch';

export const PERF_POLL_MS = 1000;
export const PERF_POLL_SLOW_MS = 5000;
export const PERF_MAX_FAILURES = 3;

class PerfStore {
	data = $state<ApiPerfResponse | null>(null);
	enabled = $state(false);
	error = $state<string | null>(null);
	unavailable = $state(false);

	private failures = 0;
	private generation = 0; // bumped on every enable: a poll started before it is answered for an older session
	private inFlight = false;
	private onVisibility = () => {
		if (document.hidden) {
			this.clearTimer();
		} else if (this.enabled && !this.unavailable && !this.inFlight) {
			this.schedule(0);
		}
	};

	private timer: ReturnType<typeof setTimeout> | null = null;

	setEnabled(on: boolean) {
		if (!browser || on === this.enabled) return;

		this.enabled = on;

		if (on) {
			this.generation++;
			this.failures = 0;
			this.error = null;
			this.unavailable = false;
			document.addEventListener('visibilitychange', this.onVisibility);

			if (!this.inFlight) this.schedule(0);
		} else {
			this.clearTimer();
			document.removeEventListener('visibilitychange', this.onVisibility);
			this.data = null;
			this.unavailable = false;
		}
	}

	private clearTimer() {
		if (this.timer !== null) {
			clearTimeout(this.timer);
			this.timer = null;
		}
	}

	private async poll() {
		if (!this.enabled || this.unavailable || document.hidden) return;

		this.inFlight = true;
		const generation = this.generation;

		try {
			const data = await PerfService.fetch();

			if (this.enabled && generation === this.generation) {
				this.data = data;
				this.error = null;
				this.failures = 0;
			}
		} catch (e) {
			if (generation !== this.generation) {
				// an answer for an earlier session says nothing about the server now
			} else if (e instanceof ApiError && e.status === 404) {
				this.unavailable = true;
				this.data = null;
			} else {
				this.failures++;
				this.error = e instanceof Error ? e.message : String(e);
			}
		} finally {
			this.inFlight = false;
		}

		if (this.enabled && !this.unavailable && !document.hidden) {
			// re-enabled while this poll was in flight: the new session gets its immediate fetch now
			if (generation !== this.generation) {
				this.schedule(0);
			} else {
				this.schedule(this.failures >= PERF_MAX_FAILURES ? PERF_POLL_SLOW_MS : PERF_POLL_MS);
			}
		}
	}

	private schedule(ms: number) {
		this.clearTimer();
		this.timer = setTimeout(() => {
			this.timer = null;
			void this.poll();
		}, ms);
	}
}

export const perfStore = new PerfStore();
