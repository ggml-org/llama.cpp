/**
 * Formatters of the performance panel, one fixed format per figure:
 * - memory: base 1024, shown as "GB", no decimals when integral or >= 100, else 1 (`241 / 251 GB`, `11.2 / 12 GB`)
 * - bandwidth: base 1000 (`4.2 GB/s` from 1e9 up, else `640 MB/s`)
 * - speed: 2 decimals below 10, else 1 (`0.53 t/s`, `14.8 t/s`)
 * - percent: integer (`87%`)
 * - unknown (null, undefined, NaN): `n/a`
 */

export const PERF_NA = 'n/a';

/** A figure that is about to be known (e.g. the speed before the second generated token). */
export const PERF_PENDING = '...';

const GIB = 1024 * 1024 * 1024;

function known(v: number | null | undefined): v is number {
	return typeof v === 'number' && Number.isFinite(v);
}

/** One memory amount in GB (base 1024), without the unit. */
export function formatPerfGbValue(bytes: number | null | undefined): string {
	if (!known(bytes)) return PERF_NA;

	const gb = bytes / GIB;

	if (gb >= 100) return Math.round(gb).toFixed(0);

	const r = Math.round(gb * 10) / 10;

	return Number.isInteger(r) ? r.toFixed(0) : r.toFixed(1);
}

/** One memory amount, e.g. `209 GB`. */
export function formatPerfGb(bytes: number | null | undefined): string {
	return known(bytes) ? `${formatPerfGbValue(bytes)} GB` : PERF_NA;
}

/** Occupied of total, e.g. `241 / 251 GB`. */
export function formatPerfMemory(
	used: number | null | undefined,
	total: number | null | undefined
): string {
	if (!known(used) || !known(total)) return PERF_NA;

	return `${formatPerfGbValue(used)} / ${formatPerfGbValue(total)} GB`;
}

/** Read bandwidth, e.g. `4.2 GB/s` or `640 MB/s`. */
export function formatPerfBandwidth(bps: number | null | undefined): string {
	if (!known(bps)) return PERF_NA;

	// round first: 999.6 MB/s is shown as 1.0 GB/s, not 1000 MB/s
	const mb = Math.round(bps / 1e6);

	if (mb >= 1000) return `${(Math.round(bps / 1e8) / 10).toFixed(1)} GB/s`;

	return `${mb.toFixed(0)} MB/s`;
}

/** Tokens per second, e.g. `0.53 t/s`. */
export function formatPerfTps(tps: number | null | undefined): string {
	if (!known(tps)) return PERF_NA;

	// round first: 9.996 is shown as 10.0, not 10.00
	const r2 = Math.round(tps * 100) / 100;

	return r2 < 10 ? `${r2.toFixed(2)} t/s` : `${(Math.round(tps * 10) / 10).toFixed(1)} t/s`;
}

/** A percentage 0..100, e.g. `87%`. */
export function formatPerfPercent(pct: number | null | undefined): string {
	if (!known(pct)) return PERF_NA;

	return `${Math.round(pct).toFixed(0)}%`;
}

/** A ratio 0..1 as a percentage, e.g. `41%`. */
export function formatPerfRatio(ratio: number | null | undefined): string {
	return known(ratio) ? formatPerfPercent(ratio * 100) : PERF_NA;
}
