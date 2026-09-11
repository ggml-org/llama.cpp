import { ModelAuxSidecar, ModelDraftSidecar, SidecarForm } from '$lib/enums';
import { HuggingFaceService } from '$lib/services/huggingface.service';
import { ModelsService } from '$lib/services/models.service';
import { describe, expect, it } from 'vitest';

const { buildDownloadTag, isSidecarEntry } = ModelsService;
const { extractQuantMeta } = HuggingFaceService;

// the sidecar filename grammar mirrors the server (common/download.cpp):
// the token must be lowercase, and it can sit at the start, between name
// segments, or at the end of the file name
describe('extractQuantMeta', () => {
	it('parses the prefix form', () => {
		expect(extractQuantMeta('mtp-Model-Q4_0.gguf')).toStrictEqual({
			quant: 'Q4_0',
			shared: false,
			sidecar: ModelDraftSidecar.MTP,
			sidecarForm: SidecarForm.PREFIX
		});
	});

	it('parses the infix form', () => {
		expect(extractQuantMeta('Model-mtp-Q4_0.gguf')).toStrictEqual({
			quant: 'Q4_0',
			shared: false,
			sidecar: ModelDraftSidecar.MTP,
			sidecarForm: SidecarForm.INFIX
		});
	});

	it('parses the suffix form', () => {
		expect(extractQuantMeta('gemma-4-E2B-it-BF16-mtp.gguf')).toStrictEqual({
			quant: 'BF16',
			shared: false,
			sidecar: ModelDraftSidecar.MTP,
			sidecarForm: SidecarForm.SUFFIX
		});
	});

	it('parses an uppercase infix token', () => {
		expect(extractQuantMeta('gemma-4-31B-it-MTP-BF16.gguf')).toStrictEqual({
			quant: 'BF16',
			shared: false,
			sidecar: ModelDraftSidecar.MTP,
			sidecarForm: SidecarForm.INFIX
		});
	});

	it('parses an uppercase trailing token', () => {
		expect(extractQuantMeta('gemma-4-E2B-it-BF16-MTP.gguf')).toStrictEqual({
			quant: 'BF16',
			shared: false,
			sidecar: ModelDraftSidecar.MTP,
			sidecarForm: SidecarForm.SUFFIX
		});
	});

	it('parses a short-form sidecar with a bare quant', () => {
		expect(extractQuantMeta('mmproj-F16.gguf')).toStrictEqual({
			quant: 'F16',
			shared: false,
			sidecar: ModelAuxSidecar.MMPROJ,
			sidecarForm: SidecarForm.PREFIX
		});
	});

	it('parses a bare sidecar file', () => {
		expect(extractQuantMeta('imatrix.gguf')).toStrictEqual({
			quant: null,
			shared: false,
			sidecar: ModelAuxSidecar.IMATRIX,
			sidecarForm: SidecarForm.PREFIX
		});
	});

	it('parses a standalone sidecar with a draft tail', () => {
		expect(extractQuantMeta('Model-mtp-draft.gguf')).toStrictEqual({
			quant: null,
			shared: false,
			sidecar: ModelDraftSidecar.MTP,
			sidecarForm: SidecarForm.SUFFIX
		});
	});

	it('parses a nested sidecar path by its file name', () => {
		expect(extractQuantMeta('MTP/mtp-Model-Q4_0.gguf')).toStrictEqual({
			quant: 'Q4_0',
			shared: false,
			sidecar: ModelDraftSidecar.MTP,
			sidecarForm: SidecarForm.PREFIX
		});
	});

	it('returns null for non-weight files', () => {
		expect(extractQuantMeta('README.md')).toBeNull();
	});
});

describe('buildDownloadTag', () => {
	it('appends the quantization', () => {
		expect(buildDownloadTag('org/repo', 'Q4_0', null)).toBe('org/repo:Q4_0');
	});

	it('appends the quantization and sidecar', () => {
		expect(buildDownloadTag('org/repo', 'Q4_0', ModelDraftSidecar.MTP)).toBe('org/repo:Q4_0-mtp');
	});

	it('uses the sidecar alone when there is no quant', () => {
		expect(buildDownloadTag('org/repo', null, ModelAuxSidecar.MMPROJ)).toBe('org/repo:mmproj');
	});

	it('returns the repo id untouched without a tag', () => {
		expect(buildDownloadTag('org/repo', null, null)).toBe('org/repo');
	});
});

describe('isSidecarEntry', () => {
	it('detects sidecar entries by their tag', () => {
		expect(isSidecarEntry('org/repo:Q4_0-mtp')).toBe(true);
		expect(isSidecarEntry('org/repo:mmproj')).toBe(true);
	});

	it('leaves plain model entries loadable', () => {
		expect(isSidecarEntry('org/repo:Q4_0')).toBe(false);
		expect(isSidecarEntry('org/repo')).toBe(false);
	});
});
