import { SETTINGS_KEYS } from '$lib/constants';
import { AttachmentType } from '$lib/enums';
import {
	getPdfParseMode,
	getPdfProcessingLabel,
	resolvePdfParseModeFromExtra
} from '$lib/utils/pdf-parse-mode';
import { describe, expect, it } from 'vitest';

describe('getPdfParseMode', () => {
	it('defaults to none', () => {
		expect(getPdfParseMode(undefined)).toBe('none');
		expect(getPdfParseMode({})).toBe('none');
	});

	it('returns image when pdfAsImage is set', () => {
		expect(
			getPdfParseMode({
				[SETTINGS_KEYS.PDF_AS_IMAGE]: true,
				[SETTINGS_KEYS.PDF_PARSE_TEXT]: true
			})
		).toBe('image');
	});

	it('returns text when parse-text is set and image is not', () => {
		expect(getPdfParseMode({ [SETTINGS_KEYS.PDF_PARSE_TEXT]: true })).toBe('text');
	});

	it('returns none when parse-none is set', () => {
		expect(getPdfParseMode({ [SETTINGS_KEYS.PDF_PARSE_NONE]: true })).toBe('none');
	});
});

describe('getPdfProcessingLabel', () => {
	it('labels a no-parse extra as original file, not text', () => {
		expect(
			getPdfProcessingLabel(
				{
					base64Data: 'JVBERi0=',
					content: '',
					name: 'doc.pdf',
					parsedAs: 'none',
					processedAsImages: false,
					type: AttachmentType.PDF
				},
				'text'
			)
		).toBe('Original file');
	});

	it('falls back to settings mode for a draft without extras', () => {
		expect(getPdfProcessingLabel(undefined, 'none')).toBe('Original file');
		expect(getPdfProcessingLabel(undefined, 'text')).toBe('Sent as Text');
		expect(getPdfProcessingLabel(undefined, 'image')).toBe('Sent as Image');
	});

	it('treats a legacy extra with extracted content as text', () => {
		expect(
			resolvePdfParseModeFromExtra({
				base64Data: 'JVBERi0=',
				content: 'hello',
				name: 'doc.pdf',
				processedAsImages: false,
				type: AttachmentType.PDF
			})
		).toBe('text');
	});
});
