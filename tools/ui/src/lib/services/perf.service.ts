/**
 * PerfService - Fetches live server performance from /perf
 *
 * No reactive state; polled by perfStore while the performance panel is shown.
 */

import { API_PERF } from '$lib/constants';
import type { ApiPerfResponse } from '$lib/types';
import { apiFetch } from '$lib/utils/api-fetch';

export class PerfService {
	/**
	 * @throws {ApiError} with status 503 while the model loads, 404 on servers without /perf
	 */
	static async fetch(): Promise<ApiPerfResponse> {
		return apiFetch<ApiPerfResponse>(API_PERF.GET, { authOnly: true });
	}
}
