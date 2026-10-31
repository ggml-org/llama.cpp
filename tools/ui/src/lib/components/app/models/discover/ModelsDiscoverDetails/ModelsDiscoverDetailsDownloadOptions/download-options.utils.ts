import { DRAFT_FILE_LABEL, MODEL_ID, PATH_SEPARATOR } from '$lib/constants';
import { ModelDraftSidecar, ModelSelectableFileKind } from '$lib/enums';
import { HuggingFaceService } from '$lib/services';
import type { ModelSelectableFile } from '$lib/types';
import { isAuxSidecar, isDraftSidecar } from '$lib/utils';

/**
 * Draft kinds in the order a chip lists them. DFlash wins: on the models that
 * ship both it drafts a block at a time and stays accurate further out than
 * MTP's few tokens; a kind not listed here sorts last.
 */
export const DRAFT_PREFERENCE_ORDER: readonly ModelDraftSidecar[] = [
	ModelDraftSidecar.DFLASH,
	ModelDraftSidecar.MTP
];

/** Draft sidecar of a file, typed for files already known to be drafts; null when it carries none. */
export function draftBadgeOf(file: Pick<ModelSelectableFile, 'path'>): ModelDraftSidecar | null {
	const sidecar = HuggingFaceService.extractQuantMeta(file.path)?.sidecar;

	return sidecar && isDraftSidecar(sidecar) ? sidecar : null;
}

/** Kind of a file path: the main weights, a draft sidecar, or an aux sidecar (mmproj). */
export function classify(path: string): ModelSelectableFileKind {
	const sidecar = HuggingFaceService.extractQuantMeta(path)?.sidecar;

	if (!sidecar) return ModelSelectableFileKind.MAIN;

	return isAuxSidecar(sidecar) ? ModelSelectableFileKind.AUX : ModelSelectableFileKind.DRAFT;
}

/** Draft kinds best first, so a chip lists its joined badges in the pull order. */
export function sortDraftKinds(kinds: readonly ModelDraftSidecar[]): ModelDraftSidecar[] {
	const rank = (kind: ModelDraftSidecar) => {
		const index = DRAFT_PREFERENCE_ORDER.indexOf(kind);

		return index === -1 ? DRAFT_PREFERENCE_ORDER.length : index;
	};

	return [...kinds].sort((a, b) => rank(a) - rank(b));
}

/** Display label of a file: its quant, else the file name without the extension. */
export function labelFor(path: string): string {
	const meta = HuggingFaceService.extractQuantMeta(path);

	if (meta?.quant) return meta.quant;

	// Quantless sidecar files: the chip badge already carries the sidecar type,
	// so the label only marks draft files; aux sidecars badge alone.
	if (meta?.sidecar) return isDraftSidecar(meta.sidecar) ? DRAFT_FILE_LABEL : '';

	const basename = path.split(PATH_SEPARATOR).pop() ?? path;

	return basename.replace(MODEL_ID.WEIGHT_EXTENSION_REGEX, '');
}
