/**
 * Model hardware-compatibility estimation, ported from ggml-org/llama-macos
 * (`Model+Compatibility.swift`, `HFRepoResolver.swift`, `SidecarPicker.swift`).
 *
 * A quant "fits" when its estimated runtime memory stays within the device
 * budget. The budget mirrors llama.cpp's own fit target: the GPU working set
 * (approximated from RAM) minus a fit slack, clamped by an OS floor so the
 * desktop keeps enough to run. We cannot read Metal's working set from a
 * browser, so the working set is approximated as 75% of RAM (Apple's ratio on
 * the machines llama-macos targets).
 *
 * Compatibility tiers:
 * - `full`     -> fits at the model's native max context (green)
 * - `limited`  -> fits only at a reduced context (yellow)
 * - `none`     -> does not fit even at the minimum context (red)
 */

import { browser } from '$app/environment';
import { HuggingFaceService } from '$lib/services';
import type { HfModelSibling } from '$lib/types/huggingface';

/** llama.cpp's default `--fit-target` margin, in MB. */
const FIT_SLACK_MB = 1024;
/** Physical RAM kept out of a model's reach for the OS and other apps, in MB. */
const OS_FLOOR_MB = 4096;
/** Overhead multiplier applied to the file size when estimating weight memory. */
const WEIGHT_OVERHEAD_MULTIPLIER = 1.05;
/** Fraction of RAM the GPU working set is approximated as (Apple's ~75%). */
const WORKING_SET_FRACTION = 0.75;
/** Minimum context a model must support to launch, matching llama.cpp's default. */
const MIN_CTX_TOKENS = 4096;
/** Standard context tiers, ascending, used to find the largest fitting one. */
const CTX_TIERS = [4096, 8192, 16384, 32768, 65536, 131072, 262144] as const;

export type CompatibilityTier = 'full' | 'limited' | 'none';

const MB = 1024 * 1024;
/** Hardcoded device RAM (GB) for testing the compatibility UI; 0 disables it. */
const TEST_DEVICE_MEMORY_GB = 128;

/**
 * Resolve the device memory in GB: the user's settings override when set,
 * else the browser's `navigator.deviceMemory` (Chrome/Edge only, capped at 8).
 * Returns 0 when neither is available, which callers treat as "unknown".
 */
export function resolveDeviceMemoryGb(configuredGb: number): number {
	// Hardcoded device RAM for testing the compatibility UI; 0 disables the
	// override. TODO: remove once the device-memory source is trusted.
	if (TEST_DEVICE_MEMORY_GB > 0) return TEST_DEVICE_MEMORY_GB;

	if (configuredGb > 0) return configuredGb;

	if (!browser) return 0;

	const nav = navigator as Navigator & { deviceMemory?: number };

	return typeof nav.deviceMemory === 'number' && nav.deviceMemory > 0 ? nav.deviceMemory : 0;
}

/**
 * Memory a model may use, in MB: whichever of the GPU working set (less the
 * fit slack) or the RAM-less-OS-floor binds first. Returns 0 when unknown.
 */
export function deviceMemoryBudgetMb(deviceMemoryGb: number): number {
	if (deviceMemoryGb <= 0) return 0;

	const physicalMb = deviceMemoryGb * 1024;
	const gpuLimbMb = physicalMb * WORKING_SET_FRACTION - FIT_SLACK_MB;
	const ramLimbMb = physicalMb - OS_FLOOR_MB;

	return Math.max(Math.min(gpuLimbMb, ramLimbMb), 0);
}

/**
 * Map every GGUF file in the repo to a compatibility tier. Main quants get
 * their own tier; their shards and sidecars (mmproj + draft head) inherit the
 * matched main quant's tier so the whole group reads consistently.
 */
export function computeFileCompatibilityTiers(
	files: HfModelSibling[],
	nativeCtxTokens: number,
	deviceMemoryGb: number
): Map<string, CompatibilityTier> {
	const tiers = new Map<string, CompatibilityTier>();

	// Unknown device memory: leave every file untiered (neutral) rather than
	// guessing a fit we cannot back up.
	if (deviceMemoryGb <= 0) return tiers;

	const allPaths = new Set(files.map((f) => f.path));
	const sizeByPath = new Map(files.map((f) => [f.path, f.size ?? 0]));
	const budgetMb = deviceMemoryBudgetMb(deviceMemoryGb);
	// Candidate mains: skip draft heads, mmproj, imatrix, and non-first shards.
	const mains = files.filter((f) => {
		const meta = HuggingFaceService.extractQuantMeta(f.path);

		if (!meta || meta.variant !== null) return false;

		if ((f.path.split('/').pop() ?? f.path).toLowerCase().includes('imatrix')) return false;

		return !isNonFirstShard(f.path);
	});
	// Track each main's quant bits + directory so draft sidecars can be matched
	// to their closest-quant main for a tier below.
	const mainInfos: MainQuantInfo[] = [];

	for (const main of mains) {
		// Aggregate size: main + shards + mmproj + quant-matched draft.
		const picked = expandShards(main.path, allPaths);
		const mmproj = pickSidecar(main.path, files, (v) => v === 'mmproj');
		const draft = pickSidecar(main.path, files, (v) => v !== null && v !== 'mmproj');

		if (mmproj) picked.push(mmproj.path);

		const mainBytes = picked.reduce((sum, p) => sum + (sizeByPath.get(p) ?? 0), 0);
		const draftBytes = draft ? (sizeByPath.get(draft.path) ?? 0) : 0;
		const tier = compatibilityTier(mainBytes + draftBytes, nativeCtxTokens, budgetMb);

		for (const p of picked) tiers.set(p, tier);

		if (draft) tiers.set(draft.path, tier);

		const meta = HuggingFaceService.extractQuantMeta(main.path);
		const bits = meta?.quant ? (HuggingFaceService.getBitDepth(meta.quant) ?? 0) : 0;

		mainInfos.push({ bits, dirs: dirComponents(main.path), tier });
	}

	// Assign every remaining draft sidecar (mtp, dflash, ...) the tier of its
	// closest-quant main, so all variant badges show a fit icon - not just the
	// single draft counted toward a main's size.
	for (const file of files) {
		if (tiers.has(file.path)) continue;

		const meta = HuggingFaceService.extractQuantMeta(file.path);

		if (!meta?.variant || meta.variant === 'mmproj') continue;

		const main = bestMainForSidecar(file.path, meta.quant, mainInfos);

		if (main) tiers.set(file.path, main.tier);
	}

	return tiers;
}

interface MainQuantInfo {
	bits: number;
	dirs: string[];
	tier: CompatibilityTier;
}

/**
 * Find the main quant a draft sidecar pairs with: among mains in the
 * sidecar's directory or a descendant of it, the one with the closest quant
 * bit depth.
 */
function bestMainForSidecar(
	sidecarPath: string,
	sidecarQuant: string | null,
	mains: MainQuantInfo[]
): MainQuantInfo | null {
	const sidecarDirs = dirComponents(sidecarPath);
	const sidecarBits = sidecarQuant ? (HuggingFaceService.getBitDepth(sidecarQuant) ?? 0) : 0;

	let best: { diff: number; main: MainQuantInfo } | null = null;

	for (const main of mains) {
		// The sidecar's directory must be the main's directory or an ancestor.
		if (sidecarDirs.length > main.dirs.length) continue;

		if (!sidecarDirs.every((d, i) => main.dirs[i] === d)) continue;

		const diff = Math.abs(main.bits - sidecarBits);

		if (!best || diff < best.diff) best = { diff, main };
	}

	return best?.main ?? null;
}

/**
 * Weight memory (MB) is the file size with overhead; context memory scales
 * with the requested window. A quant is `full` when it fits at the native max
 * context, `limited` when it only fits at a smaller standard tier, and `none`
 * when it does not fit even at the minimum context.
 */
function compatibilityTier(
	totalBytes: number,
	nativeCtxTokens: number,
	budgetMb: number
): CompatibilityTier {
	// A known-but-tiny budget (<= 0) means nothing fits; unknown memory is
	// handled by the caller returning no tiers at all.
	const weightMb = (totalBytes / MB) * WEIGHT_OVERHEAD_MULTIPLIER;
	const ctxBytesPer1k = ctxBytesPer1kTokens(nativeCtxTokens);
	const fits = (ctxTokens: number) =>
		weightMb + (ctxBytesPer1k * (ctxTokens / 1000)) / MB <= budgetMb;

	if (nativeCtxTokens < MIN_CTX_TOKENS) return 'none';

	if (fits(nativeCtxTokens)) return 'full';

	// Find the largest standard tier that still fits within the native window.
	const largestFitting = [...CTX_TIERS]
		.filter((t) => t <= nativeCtxTokens)
		.reverse()
		.find((t) => fits(t));

	return largestFitting !== undefined ? 'limited' : 'none';
}

/**
 * Approximate KV-cache bytes per 1k tokens. Without a MemProfile probe (which
 * only exists post-launch in llama-macos) we estimate from the native context
 * window; ~0.1 MB per 1k tokens is a conservative mid-range for modern models.
 */
function ctxBytesPer1kTokens(_nativeCtxTokens: number): number {
	return 0.1 * MB;
}

/** Directory components of a repo-relative path (`Q4_K_M/a.gguf` -> `["Q4_K_M"]`). */
function dirComponents(path: string): string[] {
	return path.split('/').slice(0, -1);
}

/** True for a split-shard continuation (`-00002-of-00003.gguf`), not the first shard. */
function isNonFirstShard(path: string): boolean {
	const match = /-(\d{5})-of-(\d{5})\.gguf$/i.exec(path);

	return match !== null && match[1] !== '00001';
}

/** Expand a main GGUF to its full shard set; non-sharded files return `[main]`. */
function expandShards(main: string, allPaths: Set<string>): string[] {
	const match = /-(\d{5})-of-(\d{5})\.gguf$/i.exec(main);

	if (!match) return [main];

	const total = parseInt(match[2], 10);
	const stem = main.slice(0, main.length - match[0].length);
	const shards: string[] = [];

	for (let i = 1; i <= total; i++) {
		const shard = `${stem}-${String(i).padStart(5, '0')}-of-${String(total).padStart(5, '0')}.gguf`;

		if (allPaths.has(shard)) shards.push(shard);
	}

	return shards.length > 0 ? shards : [main];
}

/**
 * Pick the best sidecar (mmproj or draft head) for a main file, mirroring
 * llama.cpp's `find_best_sibling`: the candidate's directory must be the main
 * file's directory or an ancestor; candidates rank by deepest directory, then
 * exact quant-tag match, then closest quant-bit distance.
 */
function pickSidecar(
	main: string,
	files: HfModelSibling[],
	isCandidate: (variant: string | null) => boolean
): HfModelSibling | null {
	const mainDirs = dirComponents(main);
	const mainMeta = HuggingFaceService.extractQuantMeta(main);
	const mainBits = mainMeta?.quant ? (HuggingFaceService.getBitDepth(mainMeta.quant) ?? 0) : 0;
	const mainTag = mainMeta?.quant?.toUpperCase();

	let best: { depth: number; diff: number; exact: boolean; file: HfModelSibling } | null = null;

	for (const file of files) {
		const meta = HuggingFaceService.extractQuantMeta(file.path);

		if (!meta || !isCandidate(meta.variant)) continue;

		const dirs = dirComponents(file.path);

		if (dirs.length > mainDirs.length || !dirs.every((d, i) => mainDirs[i] === d)) continue;

		const depth = dirs.length;
		const bits = meta.quant ? (HuggingFaceService.getBitDepth(meta.quant) ?? 0) : 0;
		const diff = Math.abs(bits - mainBits);
		const exact = mainTag ? file.path.toUpperCase().includes(`-${mainTag}.`) : false;

		if (best) {
			const better =
				depth > best.depth ||
				(depth === best.depth && exact && !best.exact) ||
				(depth === best.depth && exact === best.exact && diff < best.diff);

			if (!better) continue;
		}

		best = { depth, diff, exact, file };
	}

	return best?.file ?? null;
}
