/**
 * Models Discover — download options constants.
 *
 * Shared values backing the download-options area of the details pane: the
 * serve `--spec-type` mapping, the bit-depth buckets, and the standalone
 * command builder. Kept here (not component-local) so they sit with the rest
 * of the discover domain constants.
 */

import type { ModelSidecar } from '$lib/constants';
import { ModelAuxSidecar, ModelDraftSidecar } from '$lib/enums';

/**
 * `--spec-type` value for each draft sidecar; aux sidecars (mmproj, imatrix)
 * carry none and stay empty.
 */
export const SPEC_TYPE: Record<ModelSidecar, string> = {
	[ModelAuxSidecar.IMATRIX]: '',
	[ModelAuxSidecar.MMPROJ]: '',
	[ModelDraftSidecar.DFLASH]: 'draft-dflash',
	[ModelDraftSidecar.DSPARK]: 'draft-dspark',
	[ModelDraftSidecar.EAGLE3]: 'eagle3',
	[ModelDraftSidecar.MTP]: 'draft-mtp'
};

/** Bit-depth bucket for files that carry no quant token; rendered as "Other". */
export const OTHER_BIT_DEPTH = 99;

/** Label for the OTHER_BIT_DEPTH bucket. */
export const OTHER_BIT_DEPTH_LABEL = 'Other';

/** Suffix appended after a bit-depth number to label a row, e.g. `4-bit`. */
export const BIT_DEPTH_LABEL_SUFFIX = '-bit';

/**
 * Fixed tokens of the standalone `llama serve` command preview, so the command
 * builder and its rendered spans share one source of the CLI spelling.
 */
export const SERVE_COMMAND = {
	BIN: 'llama',
	/** Draft model repo:tag argument (download sidecar / draft weights). */
	DRAFT_FLAG: '-hfd',
	/** Main model repo:tag argument. */
	MODEL_FLAG: '-hf',
	/** Speculative-decoding type argument for the draft. */
	SPEC_TYPE_FLAG: '--spec-type',
	SUBCOMMAND: 'serve'
} as const;

/** Bit depth preferred for the command's default base quant. */
export const DEFAULT_BASE_BIT_DEPTH = 4;

/** Label shown for a draft-sidecar chip whose quant is unknown. */
export const DRAFT_FILE_LABEL = 'draft';
