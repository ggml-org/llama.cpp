import { DRAFT_FILE_LABEL, isAuxSidecar, isDraftSidecar, OTHER_BIT_DEPTH } from '$lib/constants';
import { SelectableFileKind } from '$lib/enums';
import { HuggingFaceService } from '$lib/services';

/** Leading `UD-` (Unsloth Dynamic) quant prefix and its length. */
const UD_QUANT_PREFIX = 'UD-';
const UD_QUANT_PREFIX_REGEX = /^UD-(?=.)/i;
/** Captures the bit-depth digits of a quant token, e.g. `Q4_K_M` -> `4`. */
const QUANT_BIT_DEPTH_REGEX = /(?:I?Q|F)(\d+)/i;
/** Trailing weight extension, stripped from a quantless file's label. */
const WEIGHT_EXTENSION_LABEL_REGEX = /\.gguf$/i;
/** Path separator between path segments. */
const PATH_SEPARATOR = '/';

/** Kind of a file path: the main weights, a draft sidecar, or an aux sidecar (mmproj). */
export function classify(path: string): SelectableFileKind {
	const sidecar = HuggingFaceService.extractQuantMeta(path)?.sidecar;

	if (!sidecar) return SelectableFileKind.MAIN;

	return isAuxSidecar(sidecar) ? SelectableFileKind.AUX : SelectableFileKind.DRAFT;
}

/** Display label of a file: its quant, else the file name without the extension. */
export function labelFor(path: string): string {
	const meta = HuggingFaceService.extractQuantMeta(path);

	if (meta?.quant) return meta.quant;

	// Quantless sidecar files: the chip badge already carries the sidecar type,
	// so the label only marks draft files; aux sidecars badge alone.
	if (meta?.sidecar) return isDraftSidecar(meta.sidecar) ? DRAFT_FILE_LABEL : '';

	const basename = path.split(PATH_SEPARATOR).pop() ?? path;

	return basename.replace(WEIGHT_EXTENSION_LABEL_REGEX, '');
}

/**
 * Bit depth of a quant token; `OTHER_BIT_DEPTH` when it carries none. The `UD-`
 * unsloth prefix is stripped before matching.
 */
export function quantBitDepth(quant: string | null): number {
	if (!quant) return OTHER_BIT_DEPTH;

	const stripped = UD_QUANT_PREFIX_REGEX.test(quant) ? quant.slice(UD_QUANT_PREFIX.length) : quant;
	const bit = stripped.match(QUANT_BIT_DEPTH_REGEX)?.[1];

	return bit ? Number(bit) : OTHER_BIT_DEPTH;
}
