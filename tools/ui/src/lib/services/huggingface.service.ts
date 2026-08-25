import { type DraftVariant, MODEL_ID } from '$lib/constants';
import type {
	HfCatalogEntry,
	HfModelDetailInfo,
	HfModelInfo,
	HfModelSearchParams,
	HfModelSibling,
	HfModelSort
} from '$lib/types/huggingface';

/** Variant flag in a GGUF filename (e.g. draft-mtp, diffusion-flash, multimodal projector). */
export type GgufVariant = DraftVariant;

/**
 * Where the `mtp` / `dflash` / `mmproj` token sits in the filename.
 * - `prefix`  sidecar file that lives next to the main weights, e.g. `mtp-Q4_0.gguf`
 * - `suffix`  embedded draft baked into the main weights, e.g. `Hy3-IQ1_M-mtp.gguf`
 */
export type GgufVariantForm = 'prefix' | 'suffix';

// Constants

export const HF_TASKS: Record<string, string> = {
	'audio-classification': 'Audio Classification',
	'audio-to-audio': 'Audio-to-Audio',
	'automatic-speech-recognition': 'Speech Recognition',
	conversational: 'Conversational',
	'depth-estimation': 'Depth Estimation',
	'feature-extraction': 'Feature Extraction',
	'fill-mask': 'Fill Mask',
	'image-classification': 'Image Classification',
	'image-feature-extraction': 'Image Feature Extraction',
	'image-segmentation': 'Image Segmentation',
	'image-text-to-text': 'Image-Text-to-Text',
	'image-to-text': 'Image-to-Text',
	'image-to-video': 'Image-to-Video',
	'object-detection': 'Object Detection',
	'question-answering': 'Question Answering',
	'reinforcement-learning': 'Reinforcement Learning',
	robotics: 'Robotics',
	'sentence-similarity': 'Sentence Similarity',
	summarization: 'Summarization',
	'text2text-generation': 'Text2Text Generation',
	'text-classification': 'Text Classification',
	'text-generation': 'Text Generation',
	'text-to-image': 'Text-to-Image',
	'text-to-speech': 'Text to Speech',
	'text-to-video': 'Text-to-Video',
	'token-classification': 'Token Classification',
	translation: 'Translation',
	'video-to-video': 'Video-to-Video',
	'voice-activity-detection': 'Voice Activity Detection',
	'zero-shot-classification': 'Zero-Shot Classification'
};

/**
 * Best-effort readable label for an HF pipeline tag. Falls back to a
 * title-cased version of the kebab-case `pipeline_tag` (e.g. `image-text-to-text`
 * becomes `Image-Text-to-Text`) when we don't have an explicit entry above.
 */
function pipelineTagLabel(tag: string): string {
	if (HF_TASKS[tag]) return HF_TASKS[tag];

	return tag
		.split('-')
		.map((part) => (part ? part[0].toUpperCase() + part.slice(1) : part))
		.join('-');
}

/**
 * Lucide icon name (string identifier, used to lazy-import the Svelte component)
 * matching the HF pipeline_tag. Used for the filter chips on the model browser.
 * Returns `null` for unknown tags so the consumer can render a generic icon.
 */
const HF_PIPELINE_ICONS: Record<string, string> = {
	'audio-classification': 'mic',
	'audio-to-audio': 'audio-lines',
	'automatic-speech-recognition': 'mic',
	conversational: 'message-circle',
	'depth-estimation': 'layers',
	'feature-extraction': 'hash',
	'fill-mask': 'replace',
	'image-classification': 'image',
	'image-feature-extraction': 'image',
	'image-segmentation': 'image',
	'image-text-to-text': 'image-plus',
	'image-to-text': 'image',
	'image-to-video': 'video',
	'object-detection': 'scan',
	'question-answering': 'help-circle',
	'sentence-similarity': 'equal',
	summarization: 'list-collapse',
	'text2text-generation': 'message-square-more',
	'text-generation': 'message-square',
	'text-to-image': 'image',
	'text-to-speech': 'volume-2',
	'text-to-video': 'video',
	translation: 'languages',
	'video-to-video': 'video',
	'voice-activity-detection': 'mic'
};

function pipelineTagIcon(tag: string): string | null {
	return HF_PIPELINE_ICONS[tag] ?? null;
}

export const HF_LIBRARIES: Record<string, string> = {
	gguf: 'GGUF',
	mlx: 'MLX',
	onnx: 'ONNX',
	safetensors: 'Safetensors',
	transformers: 'Transformers',
	vllm: 'vLLM'
};

/**
 * HuggingFaceService - Service for browsing and searching GGUF models on Hugging Face Hub
 */
export class HuggingFaceService {
	// Configuration

	/** Available library names with display labels */
	static readonly LIBRARIES: Record<string, string> = HF_LIBRARIES;
	/** Sort option display labels */
	static readonly SORT_LABELS: Record<HfModelSort, string> = {
		createdAt: 'Newest',
		downloads: 'Most Downloads',
		lastModified: 'Recently Updated',
		likes: 'Most Likes',
		trendingScore: 'Trending'
	};
	/** Available sort options */
	static readonly SORT_OPTIONS: HfModelSort[] = [
		'downloads',
		'likes',
		'trendingScore',
		'createdAt'
	];

	// Available options for filtering

	/** Available pipeline tasks with display labels */
	static readonly TASKS: Record<string, string> = HF_TASKS;

	private static readonly BASE_URL = 'https://huggingface.co/api/models';

	// Cached base model lookups keyed by repo id, so repeated selector opens
	// never re-hit the HF API for the same repo.
	private static baseModelCache = new Map<string, { org: string; name: string } | null>();

	private static baseModelPending = new Map<
		string,
		Promise<{ org: string; name: string } | null>
	>();

	private static readonly DEFAULT_LIMIT = 50;

	private static readonly MAX_LIMIT = 100;

	// GGUF Model Searching

	/**
	 * Map of quant token to its average bit-depth in bits-per-weight (bpw).
	 */
	private static readonly QUANT_BIT_DEPTH: Record<string, number> = {
		BF16: 16,
		F16: 16,
		IQ1_M: 1,
		IQ1_S: 1,
		IQ1_XS: 1,
		IQ1_XXS: 1,
		IQ2_M: 2,
		IQ2_S: 2,
		IQ2_XS: 2,
		IQ2_XXS: 2,
		IQ3_M: 3,
		IQ3_S: 3,
		IQ3_XS: 3,
		IQ3_XXS: 3,
		Q2_K: 2,
		Q2_K_M: 2,
		Q2_K_S: 2,
		Q3_K: 3,
		Q3_K_L: 3,
		Q3_K_M: 3,
		Q3_K_S: 3,
		Q4_0: 4,
		Q4_1: 4,
		Q4_K: 4,
		Q4_K_M: 4,
		Q4_K_S: 4,
		Q5_0: 5,
		Q5_1: 5,
		Q5_K: 5,
		Q5_K_M: 5,
		Q5_K_S: 5,
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
	 * `variantForm` records which side of the filename the variant token sat
	 * on so callers can render badges differently (e.g. prefix on the left of
	 * the quant label, suffix appended to it).
	 * `quant` is `null` for files that don't carry a bit-depth token
	 * (e.g. `*-BF16.gguf`); `variant` is `null` if no draft/aux flag is present.
	 * Returns `null` only when the filename doesn't end in `.gguf`.
	 */
	static extractQuantMeta(filename: string): {
		quant: string | null;
		variant: GgufVariant | null;
		variantForm: GgufVariantForm | null;
	} | null {
		if (!MODEL_ID.WEIGHT_EXTENSION_RE.test(filename)) return null;

		let source = filename.replace(MODEL_ID.WEIGHT_EXTENSION_RE, '');
		let variant: GgufVariant | null = null;
		let variantForm: GgufVariantForm | null = null;

		const prefixMatch = source.match(MODEL_ID.DRAFT_VARIANT_PREFIX_RE);

		if (prefixMatch) {
			variant = prefixMatch[1].toLowerCase() as GgufVariant;
			variantForm = 'prefix';
			source = prefixMatch[2];
		} else {
			const suffixMatch = source.match(MODEL_ID.DRAFT_VARIANT_SUFFIX_RE);

			if (suffixMatch) {
				const candidate = suffixMatch[1];
				const headSeg = candidate.split(MODEL_ID.SEGMENT_SEPARATOR).pop();

				if (headSeg && MODEL_ID.QUANTIZATION_SEGMENT_RE.test(headSeg)) {
					variant = suffixMatch[2].toLowerCase() as GgufVariant;
					variantForm = 'suffix';
					source = candidate;
				}
			}
		}

		// Scan dash-separated segments left-to-right for the first quant match.
		// - For sidecars like `mtp-Q4_0-180MB.gguf` the quant is `Q4_0`.
		// - For embedded MTP like `Hy3-IQ1_M-mtp.gguf` we have `Hy3-IQ1_M` and `IQ1_M` matches.
		// - For main files like `Llama-3-8B-Q4_K_M.gguf` we land on the trailing quant.
		const segments = source.split(MODEL_ID.SEGMENT_SEPARATOR);
		const quantSeg = segments.find((seg) => MODEL_ID.QUANTIZATION_SEGMENT_RE.test(seg));
		const quant = quantSeg ? quantSeg.toUpperCase() : null;

		return { quant, variant, variantForm };
	}

	// GGUF Model Browsing

	/**
	 * Filter raw siblings by file extension and sort by size descending.
	 */
	static filterByExtension(siblings: HfModelSibling[], ext: string): HfModelSibling[] {
		return siblings
			.filter((f) => f.path.toLowerCase().endsWith(ext.toLowerCase()) && (f.size ?? 0) > 0)
			.sort((a, b) => (b.size ?? 0) - (a.size ?? 0));
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

	/**
	 * Format a min-max size range with a single shared unit and no spaces
	 * around the dash, e.g. `19.0-28.6 GB`.
	 */
	static formatSizeRange(min: number, max: number): string {
		const unit = max >= 1_000_000_000 ? 'GB' : max >= 1_000_000 ? 'MB' : max >= 1_000 ? 'KB' : 'B';
		const div =
			unit === 'GB' ? 1_000_000_000 : unit === 'MB' ? 1_000_000 : unit === 'KB' ? 1_000 : 1;
		const fmt = (n: number) => (div === 1 ? `${n}` : `${(n / div).toFixed(1)}`);

		return `${fmt(min)}-${fmt(max)} ${unit}`;
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

	// Model Details & Files

	/**
	 * Avatar URL for an author (org or user). 404s when the author does not
	 * exist, so callers should provide a fallback.
	 */
	static getAvatarUrl(author: string): string {
		return `https://huggingface.co/api/avatars/${author}`;
	}

	/**
	 * Resolve the original (non-GGUF) base model `{ org, name }` for a GGUF repo
	 * from its HF card (`cardData.base_model`). Returns null when the card has no
	 * base model. Results are cached per repo.
	 */
	static getBaseModel(repoId: string): Promise<{ org: string; name: string } | null> {
		const cached = this.baseModelCache.get(repoId);

		if (cached !== undefined) return Promise.resolve(cached);

		const pending = this.baseModelPending.get(repoId);

		if (pending) return pending;

		const promise = (async () => {
			const details = await this.getDetails(repoId);
			const base = this.getBaseModels(details)[0];

			if (!base) return null;

			const [org, ...rest] = base.split('/');

			return { name: rest.join('/'), org };
		})();

		this.baseModelPending.set(repoId, promise);

		promise
			.then((result) => this.baseModelCache.set(repoId, result))
			.finally(() => this.baseModelPending.delete(repoId));

		return promise;
	}

	/**
	 * Extract the original (non-GGUF) base model ids for a repo, from
	 * `cardData.base_model` (string or list) and the `base_model:` tags.
	 */
	static getBaseModels(model: HfModelDetailInfo | null): string[] {
		if (!model) return [];

		const cardBase = model.cardData?.base_model;
		const fromCard: string[] = Array.isArray(cardBase) ? cardBase : cardBase ? [cardBase] : [];
		const fromTags = (model.tags ?? [])
			.map((t) => /^base_model:(?:quantized:)?(.+)$/.exec(t)?.[1])
			.filter((v): v is string => Boolean(v));

		return Array.from(new Set([...fromCard, ...fromTags]));
	}

	/**
	 * Look up the average bit-depth for a known GGUF quantization.
	 * Returns `null` for unrecognized tokens.
	 */
	static getBitDepth(quant: string): number | null {
		return HuggingFaceService.QUANT_BIT_DEPTH[quant] ?? null;
	}

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

	/**
	 * Get detailed information about a specific GGUF model
	 */
	/**
	 * Fetch the llama.app model catalog (https://llama.app/v1/catalog.json).
	 * Returns an empty array on failure so callers can fall back gracefully.
	 */
	static async getCatalog(): Promise<HfCatalogEntry[]> {
		const url = 'https://llama.app/v1/catalog.json';

		try {
			const response = await fetch(url);

			if (!response.ok) throw new Error(`Failed to fetch catalog: ${response.status}`);

			return (await response.json()) as HfCatalogEntry[];
		} catch (error) {
			console.error('Error fetching catalog:', error);

			return [];
		}
	}

	static async getDetails(modelId: string): Promise<HfModelDetailInfo | null> {
		// Do not encode the modelId, it contains slashes for author/name.
		// `full=true` includes cardData (description, base_model) and safetensors.
		const url = `https://huggingface.co/api/models/${modelId}?full=true`;

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
	 * Get model URL on Hugging Face Hub
	 */
	static getModelUrl(modelId: string): string {
		return `https://huggingface.co/${modelId}`;
	}

	// Utility Methods

	/**
	 * Get most liked GGUF models
	 */
	static async getMostLiked(
		limit: number = HuggingFaceService.DEFAULT_LIMIT
	): Promise<HfModelInfo[]> {
		return this.search({ limit, sort: 'likes' });
	}

	/**
	 * Get newly released GGUF models
	 */
	static async getNew(limit: number = HuggingFaceService.DEFAULT_LIMIT): Promise<HfModelInfo[]> {
		return this.search({ limit, sort: 'createdAt' });
	}

	/**
	 * Get most popular GGUF models by downloads
	 */
	static async getPopular(
		limit: number = HuggingFaceService.DEFAULT_LIMIT
	): Promise<HfModelInfo[]> {
		return this.search({ limit, sort: 'downloads' });
	}

	/**
	 * Fetch the raw README.md for a repo, with the YAML frontmatter stripped.
	 */
	static async getReadme(modelId: string): Promise<string | null> {
		// Do not encode the modelId, it contains slashes for author/name
		const url = `https://huggingface.co/${modelId}/raw/main/README.md`;

		try {
			const response = await fetch(url);

			if (response.status === 404) return null;

			if (!response.ok) throw new Error(`Failed to fetch README: ${response.status}`);

			return HuggingFaceService.stripFrontmatter(await response.text());
		} catch (error) {
			console.error(`Error fetching README for ${modelId}:`, error);

			return null;
		}
	}

	/**
	 * Get repository file tree to list available GGUF variants
	 */
	static async getTree(modelId: string): Promise<HfModelSibling[]> {
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
	 * Get trending GGUF models
	 */
	static async getTrending(
		limit: number = HuggingFaceService.DEFAULT_LIMIT
	): Promise<HfModelInfo[]> {
		return this.search({ limit, sort: 'trendingScore' });
	}

	/**
	 * Parse a local HF cache file path
	 * (`.../models--<org>--<name>/snapshots/<sha>/<file>`) into its repo id and
	 * repo-relative file path. Returns null when the path is not an HF cache path.
	 */
	static parseCachePath(path: string): { repo: string; file: string } | null {
		const match = /models--(.+?)\/snapshots\/[^/]+\/(.+)$/.exec(path);

		if (!match) return null;

		const parts = match[1].split('--');

		if (parts.length < 2) return null;

		return { file: match[2], repo: `${parts[0]}/${parts.slice(1).join('--')}` };
	}

	/**
	 * Best-effort parameter count parsed from a model id/name, e.g. `27B` from
	 * `Qwen3.8-27B-GGUF` or `300M` from `embeddinggemma-300M-GGUF`. Returns null
	 * when no size token is present.
	 */
	static parseParamCount(name: string): string | null {
		const match = /(?:^|[^a-z0-9])(\d+(?:[._]\d+)?)\s*([bm])(?![a-z0-9])/i.exec(name);

		if (!match) return null;

		return `${match[1]}${match[2].toUpperCase()}`;
	}

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

		return { isGated, isGguf, isSafetensors, license, tasks };
	}

	/** Resolve a pipeline_tag to a lucide icon name, or null when unknown. */
	static pipelineTagIcon(tag: string | null | undefined): string | null {
		if (!tag) return null;

		return pipelineTagIcon(tag);
	}

	/** Resolve a pipeline_tag to a human-readable label. */
	static pipelineTagLabel(tag: string | null | undefined): string | null {
		if (!tag) return null;

		return pipelineTagLabel(tag);
	}

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

	// Internal Methods

	/**
	 * Delay helper for retry logic
	 */
	private static delay(ms: number): Promise<void> {
		return new Promise((resolve) => setTimeout(resolve, ms));
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

	/** Strip a leading YAML frontmatter block (--- ... ---) from a markdown document. */
	private static stripFrontmatter(text: string): string {
		const match = text.match(/^---\r?\n[\s\S]*?\r?\n---\r?\n?/);

		return match ? text.slice(match[0].length) : text;
	}
}
