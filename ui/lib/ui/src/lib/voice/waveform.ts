import { MIN_BAR_HEIGHT } from '../constants/voice.constants.js';

function clampUnit(value: number): number {
	if (!Number.isFinite(value)) return 0;

	return Math.min(Math.max(value, 0), 1);
}

/** A clip time as m:ss (`0:07`, `1:05`, `72:30`), rounded to the nearest second. */
export function formatClipTime(ms: number): string {
	const seconds = Number.isFinite(ms) && ms > 0 ? Math.round(ms / 1000) : 0;

	return `${Math.floor(seconds / 60)}:${String(seconds % 60).padStart(2, '0')}`;
}

/** What a screen reader hears for a clip's position, e.g. `0:12 of 1:05`. */
export function formatClipPosition(elapsedMs: number, totalMs: number): string {
	return `${formatClipTime(elapsedMs)} of ${formatClipTime(totalMs)}`;
}

/** How many bars of `barWidth` with `gap` between them fit in `width` (all in px). */
export function barCapacity(width: number, barWidth: number, gap: number): number {
	if (!(width > 0) || !(barWidth > 0)) return 0;

	return Math.max(Math.floor((width + gap) / (barWidth + gap)), 0);
}

/**
 * Peaks (0..1) resampled to `count` bars: fewer bars keep the loudest peak of each bucket, so short sounds do not
 * vanish; more bars interpolate between neighbours.
 */
export function resamplePeaks(peaks: readonly number[], count: number): number[] {
	if (count <= 0) return [];
	if (peaks.length === 0) return Array.from({ length: count }, () => 0);
	if (peaks.length === count) return peaks.map(clampUnit);

	if (count < peaks.length) {
		const bars: number[] = [];

		for (let i = 0; i < count; i++) {
			const start = Math.floor((i * peaks.length) / count);
			const end = Math.max(Math.floor(((i + 1) * peaks.length) / count), start + 1);
			let peak = 0;

			for (let j = start; j < end; j++) peak = Math.max(peak, clampUnit(peaks[j]));

			bars.push(peak);
		}

		return bars;
	}

	if (peaks.length === 1) return Array.from({ length: count }, () => clampUnit(peaks[0]));

	const bars: number[] = [];

	for (let i = 0; i < count; i++) {
		const position = (i * (peaks.length - 1)) / (count - 1);
		const left = Math.floor(position);
		const right = Math.min(left + 1, peaks.length - 1);
		const t = position - left;

		bars.push(clampUnit(peaks[left]) * (1 - t) + clampUnit(peaks[right]) * t);
	}

	return bars;
}

/**
 * `count` bars of the live levels, newest last, always filling the width: a recording with fewer levels than
 * bars is stretched over all of them, a longer one shows its last `count` levels. Before the first level every
 * bar is `null` (drawn as the resting dot).
 */
export function liveBars(levels: readonly number[], count: number): (number | null)[] {
	if (count <= 0) return [];
	if (levels.length === 0) return Array.from({ length: count }, () => null);
	if (levels.length < count) return resamplePeaks(levels, count);

	return levels.slice(-count).map(clampUnit);
}

/** A bar's height in percent of the waveform's, never under `MIN_BAR_HEIGHT`. */
export function barHeightPercent(level: number | null): number {
	return Math.max(clampUnit(level ?? 0), MIN_BAR_HEIGHT) * 100;
}

/** How far through the clip playback is, 0..1. */
export function progressFraction(elapsedMs: number, totalMs: number): number {
	if (!(totalMs > 0)) return 0;

	return clampUnit(elapsedMs / totalMs);
}

/** Where a pointer at `clientX` falls along a track starting at `left`, 0..1. */
export function pointerFraction(clientX: number, left: number, width: number): number {
	if (!(width > 0)) return 0;

	return clampUnit((clientX - left) / width);
}

/** A position moved by `deltaMs`, kept inside the clip. */
export function stepPosition(elapsedMs: number, deltaMs: number, totalMs: number): number {
	if (!(totalMs > 0)) return 0;

	return Math.min(Math.max(elapsedMs + deltaMs, 0), totalMs);
}
