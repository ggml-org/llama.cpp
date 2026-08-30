import type { HfModelDetailInfo, HfModelInfo, HfModelSibling } from '$lib/types';

/**
 * Fixtures for the Models Discover stories. Shapes mirror the HF API
 * responses (`/api/models`, `/api/models/{id}?full=true`, `/tree`) with
 * realistic ggml-org style values.
 */

export const mockSiblings: HfModelSibling[] = [
	{ path: 'Q8_0/gemma-4-12b-it-Q8_0.gguf', size: 13_100_000_000 },
	{ path: 'Q8_0/mmproj-Q8_0.gguf', size: 800_000_000 },
	{ path: 'Q4_K_M/gemma-4-12b-it-Q4_K_M.gguf', size: 7_300_000_000 },
	{ path: 'Q4_K_M/mmproj-Q4_K_M.gguf', size: 530_000_000 },
	{ path: 'Q4_K_M/mtp-Q4_K_M.gguf', size: 460_000_000 },
	{ path: 'BF16/gemma-4-12b-it-BF16.gguf', size: 24_500_000_000 }
];

export const mockGemma: HfModelInfo = {
	_id: 'gemma',
	author: 'ggml-org',
	createdAt: '2026-07-01T00:00:00.000Z',
	downloads: 1_234_567,
	gguf: {
		architecture: 'gemma4',
		context_length: 131072,
		total: 12_000_000_000
	},
	id: 'ggml-org/gemma-4-12b-it-GGUF',
	library_name: 'transformers',
	likes: 8900,
	modelId: 'ggml-org/gemma-4-12b-it-GGUF',
	pipeline_tag: 'image-text-to-text',
	private: false,
	siblings: mockSiblings.map((s) => ({ rfilename: s.path })),
	tags: ['gguf', 'license:gemma', 'base_model:google/gemma-4-12b-it'],
	trendingScore: 42
};

export const mockQwen: HfModelInfo = {
	...mockGemma,
	_id: 'qwen',
	createdAt: '2026-08-15T00:00:00.000Z',
	downloads: 45600,
	gguf: {
		architecture: 'qwen3',
		context_length: 262144,
		total: 27_000_000_000
	},
	id: 'ggml-org/Qwen3.8-27B-GGUF',
	likes: 1200,
	modelId: 'ggml-org/Qwen3.8-27B-GGUF',
	pipeline_tag: 'text-generation',
	tags: ['gguf', 'license:apache-2.0'],
	trendingScore: 90
};

export const mockListModels: HfModelInfo[] = [mockGemma, mockQwen];

// Chat template exercising the tool-use and thinking detectors.
const CHAT_TEMPLATE = [
	'{%- for message in messages %}',
	'{%- if tools %}{{ tools }}{% endif %}',
	'{%- if enable_think %} {% endif %}',
	'{%- endfor %}'
].join('\n');

export const mockDetails: HfModelDetailInfo = {
	...mockGemma,
	gated: false,
	gguf: {
		architecture: 'gemma4',
		chat_template: CHAT_TEMPLATE,
		context_length: 131072,
		total: 12_000_000_000
	},
	id: 'ggml-org/gemma-4-12b-it-GGUF',
	lastModified: '2026-08-01T00:00:00.000Z',
	usedStorage: 26_000_000_000
};
