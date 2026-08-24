import { SETTINGS_KEYS } from '$lib/constants';
import { AttachmentType } from '$lib/enums';
import type { DatabaseMessageExtra, DatabaseMessageExtraPdfFile } from '$lib/types/database';

export type PdfParseMode = 'none' | 'text' | 'image';

const PDF_PARSE_LABEL: Record<PdfParseMode, string> = {
	image: 'Sent as Image',
	none: 'Original file',
	text: 'Sent as Text'
};

/**
 * Resolve how attached PDFs should be prepared.
 * Image wins if the legacy `pdfAsImage` checkbox (now a radio option) is set.
 */
export function getPdfParseMode(config: Record<string, unknown> | undefined): PdfParseMode {
	if (!config) {
		return 'none';
	}

	if (config[SETTINGS_KEYS.PDF_AS_IMAGE]) {
		return 'image';
	}

	if (config[SETTINGS_KEYS.PDF_PARSE_TEXT]) {
		return 'text';
	}

	return 'none';
}

/** Infer how a stored PDF extra was prepared. */
export function resolvePdfParseModeFromExtra(extra: DatabaseMessageExtraPdfFile): PdfParseMode {
	if (extra.parsedAs) {
		return extra.parsedAs;
	}

	if (extra.processedAsImages) {
		return 'image';
	}

	return extra.content ? 'text' : 'none';
}

export function getPdfProcessingLabel(
	attachment: DatabaseMessageExtra | undefined,
	settingsParseMode: PdfParseMode
): string | null {
	if (attachment?.type === AttachmentType.PDF) {
		return PDF_PARSE_LABEL[resolvePdfParseModeFromExtra(attachment)];
	}

	return PDF_PARSE_LABEL[settingsParseMode];
}
