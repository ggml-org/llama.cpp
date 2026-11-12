// What the variations work out from the catalog, the way llama.app does today (src/lib/catalog.ts).
import type { Build, Family, Size } from './catalog';
import { minMemoryTierGb, type ModelModalities, type ModelSizeRange, summarizeSizes } from '$lib/models';

/** The memory Macs ship with: llama.app answers "will it run on my Mac?" against these. */
export const MAC_MEM_TIERS = [8, 16, 24, 32, 48, 64, 96, 128, 192, 256, 512, 768, 1024];

/** Memory (GB) a build needs, from the bytes that load (`memBytes`) when the build has them. */
export function memFor(build: Build): number | null {
	return minMemoryTierGb(build.memBytes ?? build.sizeBytes ?? 0, MAC_MEM_TIERS);
}

/** The family's entry bar: the least memory any of its builds needs. */
export function familyMem(family: Family): number | null {
	const tiers = family.sizes
		.flatMap((size) => size.builds)
		.map(memFor)
		.filter((n): n is number => n !== null);

	return tiers.length ? Math.min(...tiers) : null;
}

export function sizeRange(sizes: Size[]): ModelSizeRange | null {
	const bytes = sizes.flatMap((size) => size.builds.map((build) => build.sizeBytes ?? 0)).filter(Boolean);

	return bytes.length ? { max: Math.max(...bytes), min: Math.min(...bytes) } : null;
}

export function modalitiesOf(sizes: Size[]): ModelModalities {
	return {
		audio: sizes.some((size) => size.capabilities?.audio),
		video: false,
		vision: sizes.some((size) => size.capabilities?.vision)
	};
}

export const reasons = (sizes: Size[]) => sizes.some((size) => size.capabilities?.reasoning);
export const usesTools = (sizes: Size[]) => sizes.some((size) => size.capabilities?.tools);

/** Bit depth of a quant, for grouping into rows: Q4_K_M and MXFP4 are 4, Q8_0 is 8, F16 is 16. */
export function bitDepthOf(quant = ''): number {
	if (/^b?f16$/i.test(quant)) return 16;

	const match = quant.match(/(?:^|_)(?:i?q|mxfp)(\d)/i);

	return match ? Number(match[1]) : 99;
}

/** "Aug 2025" from "2025-08". */
export function releasedFor(family: Family): string {
	const [year, month] = family.released.split('-').map(Number);

	return new Date(year, month - 1).toLocaleString('en', { month: 'short', year: 'numeric' });
}


export const deeplink = (build: Build) =>
	`llama://install?${new URLSearchParams({ repo: build.repo, ...(build.quant ? { quant: build.quant } : {}) })}`;

/** A size's parameter count alone, e.g. `3B` for "3B Reasoning", `26B-A4B` as is. */
export function paramsOf(size: Size): string {
	return size.params?.match(/^E?[\d.]+[BM](?:-A[\d.]+B)?/)?.[0] ?? size.params ?? size.name;
}

/** A family's sizes in one line, e.g. `E2B–31B · 5 sizes`, with Model UI's `summarizeSizes`. */
export function sizesSummary(family: Family): string {
	return summarizeSizes(family.sizes.map(paramsOf));
}
