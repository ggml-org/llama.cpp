import type {
	HfModelInfo,
	HfModelDetailInfo,
	HfModelSearchParams,
	HfModelSort
} from '$lib/types/huggingface';

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
		createdAt: 'Newest'
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
	static async getTree(modelId: string): Promise<{ path: string; size: number }[]> {
		// FIX: Do not encode the modelId
		const url = `https://huggingface.co/api/models/${modelId}/tree/main`;
		try {
			const response = await fetch(url);
			if (!response.ok) return [];
			const data = await response.json();
			return data.filter((f: { path: string; size: number }) => f.path.endsWith('.gguf'));
		} catch {
			return [];
		}
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
