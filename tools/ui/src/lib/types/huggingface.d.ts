/**
 * HuggingFace Hub Model Browsing Types
 *
 * Types for the HuggingFace REST API (/api/models)
 * Reference: https://huggingface.co/docs/huggingface_hub/package_reference/hf_api
 */

// Search Options

export interface HfModelSearchParams {
	/** Full-text search query */
	search?: string;
	/** Filter by pipeline task (e.g., "text-generation", "image-generation") */
	pipeline_tag?: string;
	/** Filter by library (e.g., "transformers", "diffusers", "gguf") */
	library_name?: string;
	/** Filter by tag (e.g., "gguf") */
	filter?: string;
	/** Filter by author or organization */
	author?: string;
	/** Sort field */
	sort?: HfModelSort;
	/** Results per page (1-100) */
	limit?: number;
	/** Pagination offset */
	offset?: number;
	/** Filter by model config */
	config?: string;
	/** Return full model info */
	full?: boolean;
	/** Filter by visibility */
	private?: boolean;
	/** Filter by gated status */
	gated?: boolean;
}

export type HfModelSort = 'downloads' | 'likes' | 'createdAt' | 'lastModified' | 'trendingScore';

// Model Info (from /api/models)

export interface HfModelInfo {
	/** Unique document ID */
	_id: string;
	/** Model ID (e.g., "meta-llama/Llama-3.1-8B-Instruct") */
	id: string;
	/** Number of likes */
	likes: number;
	/** Trending score */
	trendingScore: number;
	/** Whether the model is private */
	private: boolean;
	/** Number of downloads */
	downloads: number;
	/** Model tags */
	tags: string[];
	/** Pipeline task (e.g., "text-generation") */
	pipeline_tag: string | null;
	/** Library name (e.g., "transformers", "diffusers") */
	library_name: string | null;
	/** Creation timestamp */
	createdAt: string;
	/** Model ID (alias for id) */
	modelId: string;
}

// Model Details (with full=true)

export interface HfModelCardData {
	/** License identifier */
	license?: string;
	/** Model description */
	description?: string;
	/** Model library */
	language?: string[];
	/** Tags */
	tags?: string[];
	[key: string]: unknown;
}

export interface HfModelDetails {
	/** Model ID */
	id?: string;
	/** SHA256 digest */
	sha?: string;
	/** Last modified timestamp */
	lastModified?: string;
	/** Downloads count */
	downloads?: number;
	/** Number of likes */
	likes?: number;
	/** Whether the model is gated */
	gated?: boolean;
	/** Model card data */
	cardData?: HfModelCardData;
	/** Tags */
	tags?: string[];
	/** Pipeline tag */
	pipeline_tag?: string | null;
	/** Library name */
	library_name?: string | null;
	/** Safe tensors info */
	safetensors?: Record<string, unknown>;
	/** Model size in bytes */
	size?: number;
	[key: string]: unknown;
}

export interface HfModelDetailInfo extends HfModelInfo {
	/** Whether the model is gated (true/false/'auto') */
	gated?: boolean | string;
	/** Repository file listing mirrors of /api/models/{id}/tree/main */
	siblings?: HfModelSibling[];
	/** Author / organization (sometimes returned alongside siblings) */
	author?: string;
	/** Detailed model information (only present when full=true) */
	details?: HfModelDetails;
}

/** A single entry in a model repository's file tree */
export interface HfModelSibling {
	/** Relative path of the file or directory within the repo */
	path: string;
	/** Blob SHA, if applicable */
	rfilename?: string;
	/** Size in bytes (omitted for directories) */
	size?: number;
	/** Whether this entry is a directory */
	type?: 'file' | 'directory';
	/** OID/hash for the blob */
	oid?: string;
	[key: string]: unknown;
}

// API Response

export interface HfModelApiResponse {
	/** List of models */
	data: HfModelInfo[];
	/** Total count (if available) */
	total?: number;
}

// Task & Library Categories (runtime data lives in huggingface.ts)

export interface HfTaskCategory {
	id: string;
	label: string;
	icon: string;
	tasks: string[];
}

/**
 * Pipeline task labels mapping
 * Runtime implementation lives in `$lib/services/huggingface.ts`
 */
export const HF_TASKS: Record<string, string>;

/**
 * Library name labels mapping
 * Runtime implementation lives in `$lib/services/huggingface.ts`
 */
export const HF_LIBRARIES: Record<string, string>;
