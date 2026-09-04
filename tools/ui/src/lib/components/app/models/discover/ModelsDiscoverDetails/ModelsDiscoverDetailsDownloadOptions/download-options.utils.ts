import {
	DRAFT_FILE_LABEL,
	isAuxSidecar,
	isDraftSidecar,
	MODEL_ID,
	PATH_SEPARATOR
} from '$lib/constants';
import { ModelSelectableFileKind } from '$lib/enums';
import { HuggingFaceService } from '$lib/services';

/** Kind of a file path: the main weights, a draft sidecar, or an aux sidecar (mmproj). */
export function classify(path: string): ModelSelectableFileKind {
	const sidecar = HuggingFaceService.extractQuantMeta(path)?.sidecar;

	if (!sidecar) return ModelSelectableFileKind.MAIN;

	return isAuxSidecar(sidecar) ? ModelSelectableFileKind.AUX : ModelSelectableFileKind.DRAFT;
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
