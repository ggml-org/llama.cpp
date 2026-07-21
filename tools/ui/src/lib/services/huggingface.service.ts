import type {
	HfModelInfo,
	HfModelDetailInfo,
	HfModelSearchParams,
	HfModelSort,
	HfModelSibling
} from '$lib/types/huggingface';
import type { DraftVariant } from '$lib/constants/model-id';
import {
	DRAFT_VARIANT_PREFIX_RE,
	DRAFT_MTP_SUFFIX_RE,
	MODEL_QUANTIZATION_SEGMENT_RE,
	MODEL_WEIGHT_EXTENSION_RE,
	MODEL_ID_SEGMENT_SEPARATOR
} from '$lib/constants';

/** Variant flag in a GGUF filename (e.g. draft-mtp, diffusion-flash, multimodal projector). */
export type GgufVariant = DraftVariant;

// Constants

export const HF_TASKS: Record<string, string> = {
	'text-generation': 'Text Generation',
	conversational: 'Conversational',
	'text2text-generation': 'Text2Text Generation',
	'fill-mask': 'Fill Mask',
	'automatic-speech-recognition': 'Speech Recognition',
	'text-to-speech': 'Text to Speech',
	'sentence-similarity': 'Sentence Similarity'
};

export const HF_LIBRARIES: Record<string, string> = {
	transformers: 'Transformers',
	gguf: 'GGUF',
	safetensors: 'Safetensors',
	onnx: 'ONNX',
	vllm: 'vLLM',
	mlx: 'MLX'
};

/**
 * HuggingFaceService - Service for browsing and searching GGUF models on Hugging Face Hub
 */
export class HuggingFaceService {
	// Configuration

	private static readonly BASE_URL = 'https://huggingface.co/api/models';
	private static readonly DEFAULT_LIMIT = 50;
	private static readonly MAX_LIMIT = 100;

	// Available options for filtering

	/** Available pipeline tasks with display labels */
	static readonly TASKS: Record<string, string> = HF_TASKS;

	/** Available library names with display labels */
	static readonly LIBRARIES: Record<string, string> = HF_LIBRARIES;

	/** Available sort options */
	static readonly SORT_OPTIONS: HfModelSort[] = [
		'downloads',
		'likes',
		'trendingScore',
		'createdAt'
	];

	/** Sort option display labels */
	static readonly SORT_LABELS: Record<HfModelSort, string> = {
		downloads: 'Most Downloads',
		likes: 'Most Likes',
		trendingScore: 'Trending',
		createdAt: 'Newest',
		lastModified: 'Recently Updated'
	};

	// GGUF Model Searching

	/**
	 * Search GGUF models with various filters and options
	 */
	static async search(params: HfModelSearchParams = {}): Promise<HfModelInfo[]> {
		const { limit = HuggingFaceService.DEFAULT_LIMIT, ...restParams } = params;

		const url = this.buildUrl({
			...restParams,
			filter: 'gguf',
			limit: Math.min(limit, HuggingFaceService.MAX_LIMIT)
		});

		return this.fetchWithRetry(url);
	}

	/**
	 * Search models by query string
	 */
	static async searchByQuery(
		query: string,
		params: Omit<HfModelSearchParams, 'search'> = {}
	): Promise<HfModelInfo[]> {
		return this.search({
			...params,
			search: query
		});
	}

	// GGUF Model Browsing

	/**
	 * Get trending GGUF models
	 */
	static async getTrending(
		limit: number = HuggingFaceService.DEFAULT_LIMIT
	): Promise<HfModelInfo[]> {
		return this.search({ sort: 'trendingScore', limit });
	}

	/**
	 * Get most popular GGUF models by downloads
	 */
	static async getPopular(
		limit: number = HuggingFaceService.DEFAULT_LIMIT
	): Promise<HfModelInfo[]> {
		return this.search({ sort: 'downloads', limit });
	}

	/**
	 * Get most liked GGUF models
	 */
	static async getMostLiked(
		limit: number = HuggingFaceService.DEFAULT_LIMIT
	): Promise<HfModelInfo[]> {
		return this.search({ sort: 'likes', limit });
	}

	/**
	 * Get newly released GGUF models
	 */
	static async getNew(limit: number = HuggingFaceService.DEFAULT_LIMIT): Promise<HfModelInfo[]> {
		return this.search({ sort: 'createdAt', limit });
	}

	// GGUF Model Filtering

	/**
	 * Get GGUF models by pipeline task
	 */
	static async getByTask(
		pipelineTag: string,
		params: Omit<HfModelSearchParams, 'pipeline_tag'> = {}
	): Promise<HfModelInfo[]> {
		return this.search({
			...params,
			pipeline_tag: pipelineTag
		});
	}

	// Model Details & Files

	/**
	 * Get detailed information about a specific GGUF model
	 */
	static async getDetails(modelId: string): Promise<HfModelDetailInfo | null> {
		// FIX: Do not encode the modelId, as it contains slashes for author/name
		const url = `https://huggingface.co/api/models/${modelId}`;
		try {
			const response = await fetch(url);
			if (response.status === 404) return null;
			if (!response.ok) throw new Error(`Failed to fetch model details: ${response.status}`);
			const data = (await response.json()) as HfModelDetailInfo;
			return data;
		} catch (error) {
			console.error(`Error fetching details for ${modelId}:`, error);
			return null;
		}
	}

	/**
	 * Get repository file tree to list available GGUF variants
	 */
	static async getTree(modelId: string): Promise<HfModelSibling[]> {
		// FIX: Do not encode the modelId
		const url = `https://huggingface.co/api/models/${modelId}/tree/main`;
		try {
			const response = await fetch(url);
			if (!response.ok) return [];
			const data = (await response.json()) as HfModelSibling[];
			return data.filter((f) => f.type !== 'directory');
		} catch {
			return [];
		}
	}

	/**
	 * Filter raw siblings by file extension and sort by size descending.
	 */
	static filterByExtension(siblings: HfModelSibling[], ext: string): HfModelSibling[] {
		return siblings
			.filter((f) => f.path.toLowerCase().endsWith(ext.toLowerCase()) && (f.size ?? 0) > 0)
			.sort((a, b) => (b.size ?? 0) - (a.size ?? 0));
	}

	/**
	 * Fetch raw README.md, with YAML frontmatter stripped.
	 */
	static async getReadme(modelId: string): Promise<string | null> {
		// FIX: Do not encode the modelId
		const url = `https://huggingface.co/${modelId}/raw/main/README.md`;
		try {
			const response = await fetch(url);
			if (response.status === 404) return null;
			if (!response.ok) throw new Error(`Failed to fetch README: ${response.status}`);
			const text = await response.text();
			return HuggingFaceService.stripFrontmatter(text);
		} catch (error) {
			console.error(`Error fetching README for ${modelId}:`, error);
			return null;
		}
	}

	/**
	 * Strip a leading YAML frontmatter block (--- ... ---) from a markdown document.
	 */
	private static stripFrontmatter(text: string): string {
		// Match `---` at start, then any content (including newlines), then closing `---`.
		const match = text.match(/^---\r?\n[\s\S]*?\r?\n---\r?\n?/);
		return match ? text.slice(match[0].length) : text;
	}

	/**
	 * Map of quant token to its average bit-depth in bits-per-weight (bpw).
	 */
	private static readonly QUANT_BIT_DEPTH: Record<string, number> = {
		IQ1_XXS: 1,
		IQ1_XS: 1,
		IQ1_S: 1,
		IQ1_M: 1,
		IQ2_XXS: 2,
		IQ2_XS: 2,
		IQ2_S: 2,
		IQ2_M: 2,
		Q2_K: 2,
		Q2_K_S: 2,
		Q2_K_M: 2,
		IQ3_XXS: 3,
		IQ3_XS: 3,
		IQ3_S: 3,
		IQ3_M: 3,
		Q3_K: 3,
		Q3_K_S: 3,
		Q3_K_M: 3,
		Q3_K_L: 3,
		Q4_0: 4,
		Q4_1: 4,
		Q4_K: 4,
		Q4_K_S: 4,
		Q4_K_M: 4,
		Q5_0: 5,
		Q5_1: 5,
		Q5_K: 5,
		Q5_K_S: 5,
		Q5_K_M: 5,
		Q6_K: 6,
		Q8_0: 8
	};

	/**
	 * Extract the GGUF quantization token (e.g. `Q4_K_M`) and any draft/aux variant
	 * (`mtp`, `dflash`, `mmproj`) from a `.gguf` filename. The variant shows up
	 * either as a sidecar prefix (`mtp-<name>.gguf`, `dflash-<name>.gguf`,
	 * `mmproj-<name>.gguf`) or as the `-mtp` suffix when the draft model is
	 * embedded in the same GGUF weight file.
	 *
	 * `quant` is `null` for files that don't carry a bit-depth token
	 * (e.g. `*-BF16.gguf`); `variant` is `null` if no draft/aux flag is present.
	 * Returns `null` only when the filename doesn't end in `.gguf`.
	 */
	static extractQuantMeta(
		filename: string
	): { quant: string | null; variant: GgufVariant | null } | null {
		if (!MODEL_WEIGHT_EXTENSION_RE.test(filename)) return null;

		let source = filename.replace(MODEL_WEIGHT_EXTENSION_RE, '');
		let variant: GgufVariant | null = null;

		const prefixMatch = source.match(DRAFT_VARIANT_PREFIX_RE);
		if (prefixMatch) {
			variant = prefixMatch[1].toLowerCase() as GgufVariant;
			source = prefixMatch[2];
		} else {
			const suffixMatch = source.match(DRAFT_MTP_SUFFIX_RE);
			if (suffixMatch) {
				const candidate = suffixMatch[1];
				const headSeg = candidate.split(MODEL_ID_SEGMENT_SEPARATOR).pop();
				if (headSeg && MODEL_QUANTIZATION_SEGMENT_RE.test(headSeg)) {
					variant = 'mtp';
					source = candidate;
				}
			}
		}

		const segments = source.split(MODEL_ID_SEGMENT_SEPARATOR);
		const last = segments[segments.length - 1];
		const quant = MODEL_QUANTIZATION_SEGMENT_RE.test(last) ? last.toUpperCase() : null;

		return { quant, variant };
	}

	/**
	 * Look up the average bit-depth for a known GGUF quantization.
	 * Returns `null` for unrecognized tokens.
	 */
	static getBitDepth(quant: string): number | null {
		return HuggingFaceService.QUANT_BIT_DEPTH[quant] ?? null;
	}

	/**
	 * Resolve base-model ids referenced by this model card.
	 *
	 * Sources, merged and deduped:
	 *   - `cardData.base_model` (string or array, may be undefined)
	 *   - `base_model:*` / `base_model:quantized:*` tags
	 */
	static getBaseModels(model: HfModelDetailInfo | null): string[] {
		if (!model) return [];

		const cardBaseRaw = (model.details?.cardData as { base_model?: string | string[] } | undefined)
			?.base_model;
		const fromCard: string[] = Array.isArray(cardBaseRaw)
			? cardBaseRaw
			: cardBaseRaw
				? [cardBaseRaw]
				: [];

		const fromTags = (model.tags ?? [])
			.map((t) => /^base_model:(?:quantized:)?(.+)$/.exec(t)?.[1])
			.filter((v): v is string => Boolean(v));

		return Array.from(new Set([...fromCard, ...fromTags]));
	}

	// Model Navigation

	/**
	 * Get model URL on Hugging Face Hub
	 */
	static getModelUrl(modelId: string): string {
		return `https://huggingface.co/${modelId}`;
	}

	// Utility Methods

	/**
	 * Parse model tags to extract useful information
	 */
	static parseTags(tags: string[]): {
		license: string | null;
		isGated: boolean;
		isGguf: boolean;
		isSafetensors: boolean;
		tasks: string[];
	} {
		const license = tags.find((tag) => tag.startsWith('license:'))?.replace('license:', '') || null;
		const isGated = tags.includes('gated');
		const isGguf = tags.includes('gguf');
		const isSafetensors = tags.includes('safetensors');
		const tasks = tags.filter((tag) => Object.keys(HuggingFaceService.TASKS).includes(tag));

		return { license, isGated, isGguf, isSafetensors, tasks };
	}

	/**
	 * Format model downloads count with K/M/B suffix
	 */
	static formatDownloads(downloads: number): string {
		if (downloads >= 1_000_000) {
			return `${(downloads / 1_000_000).toFixed(1)}M`;
		}
		if (downloads >= 1_000) {
			return `${(downloads / 1_000).toFixed(1)}K`;
		}
		return downloads.toString();
	}

	/**
	 * Format likes count with K suffix if applicable
	 */
	static formatLikes(likes: number): string {
		if (likes >= 1_000) {
			return `${(likes / 1_000).toFixed(1)}K`;
		}
		return likes.toString();
	}

	/**
	 * Format timestamp to relative time
	 */
	static formatRelativeTime(timestamp: string): string {
		const date = new Date(timestamp);
		const now = new Date();
		const diffMs = now.getTime() - date.getTime();
		const diffDays = Math.floor(diffMs / (1000 * 60 * 60 * 24));

		if (diffDays === 0) return 'Today';
		if (diffDays === 1) return 'Yesterday';
		if (diffDays < 7) return `${diffDays} days ago`;
		if (diffDays < 30) return `${Math.floor(diffDays / 7)} weeks ago`;
		if (diffDays < 365) return `${Math.floor(diffDays / 30)} months ago`;
		return `${Math.floor(diffDays / 365)} years ago`;
	}

	/**
	 * Format file size in bytes to human-readable string
	 */
	static formatFileSize(bytes: number): string {
		if (bytes >= 1_000_000_000) {
			return `${(bytes / 1_000_000_000).toFixed(1)} GB`;
		}
		if (bytes >= 1_000_000) {
			return `${(bytes / 1_000_000).toFixed(1)} MB`;
		}
		if (bytes >= 1_000) {
			return `${(bytes / 1_000).toFixed(1)} KB`;
		}
		return `${bytes} B`;
	}

	// Internal Methods

	/**
	 * Build API URL from search parameters
	 */
	private static buildUrl(params: HfModelSearchParams): string {
		const url = new URL(this.BASE_URL);

		Object.entries(params).forEach(([key, value]) => {
			if (value !== undefined && value !== null && value !== '') {
				if (Array.isArray(value)) {
					value.forEach((v) => url.searchParams.append(key, v));
				} else {
					url.searchParams.set(key, String(value));
				}
			}
		});

		return url.toString();
	}

	/**
	 * Fetch data with retry logic for resilience
	 */
	private static async fetchWithRetry(url: string, attempt: number = 1): Promise<HfModelInfo[]> {
		const RETRY_ATTEMPTS = 3;
		const RETRY_DELAY_MS = 1000;

		try {
			const response = await fetch(url);

			if (!response.ok) {
				if (response.status === 404) {
					return [];
				}

				if (response.status >= 500 && attempt < RETRY_ATTEMPTS) {
					await this.delay(RETRY_DELAY_MS * attempt);
					return this.fetchWithRetry(url, attempt + 1);
				}

				throw new Error(`API request failed: ${response.status} ${response.statusText}`);
			}

			const data = await response.json();

			if (Array.isArray(data)) {
				return data as HfModelInfo[];
			}

			if (data && Array.isArray(data.data)) {
				return data.data as HfModelInfo[];
			}

			throw new Error('Unexpected API response format');
		} catch (error) {
			if (attempt < RETRY_ATTEMPTS) {
				await this.delay(RETRY_DELAY_MS * attempt);
				return this.fetchWithRetry(url, attempt + 1);
			}

			throw error;
		}
	}

	/**
	 * Delay helper for retry logic
	 */
	private static delay(ms: number): Promise<void> {
		return new Promise((resolve) => setTimeout(resolve, ms));
	}
}
