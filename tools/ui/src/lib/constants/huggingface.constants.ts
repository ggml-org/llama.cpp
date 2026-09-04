/**
 * HuggingFace Hub constants.
 *
 * URLs, parsing regexes and formatting units for the HuggingFaceService.
 * Reference: https://huggingface.co/docs/huggingface_hub/package_reference/hf_api
 */
// LLAMA-APP-REUSE: HF Hub endpoints and filename conventions

// API endpoints

export const HF_BASE_URL = 'https://huggingface.co';
export const HF_API_MODELS_URL = `${HF_BASE_URL}/api/models`;
export const HF_AVATARS_URL = `${HF_BASE_URL}/api/avatars`;

// Query params

export const HF_FULL_DETAIL_PARAM = 'full=true';
export const HF_RECURSIVE_TREE_PARAM = 'recursive=true';
/** Search filter that restricts results to repos containing GGUF files. */
export const HF_GGUF_FILTER = 'gguf';
/** Repeatable `expand` query param selecting fields on the list endpoint. */
export const HF_EXPAND_PARAM = 'expand';
/**
 * Fields the model list endpoint omits by default but the discover list rows
 * render: `gguf` (chat template, context length, param count) drives the
 * reasoning / tool-use icons and the context badge, `siblings` the vision and
 * draft-sidecar badges. Without them those parts of a row stay empty.
 */
export const HF_MODEL_LIST_EXPAND: readonly string[] = [
	'author',
	'downloads',
	'gguf',
	'lastModified',
	'likes',
	'pipeline_tag',
	'siblings',
	// `base_model:` tags, so search rows can show the base org's avatar as the
	// main avatar with the quant org as the corner badge, like catalog rows.
	'tags'
];

// Repo file conventions

export const HF_MAIN_BRANCH = 'main';
export const HF_README_FILENAME = 'README.md';
export const HF_RAW_PATH = 'raw';
export const HF_TREE_PATH = 'tree';

// Pagination

export const HF_LINK_NEXT_REGEX = /<([^>]+)>;\s*rel="next"/;
/** `Link` response header carrying the next page URL for cursor pagination. */
export const HF_LINK_HEADER = 'Link';

// Fetch retry

export const HF_RETRY_ATTEMPTS = 3;
export const HF_RETRY_DELAY_MS = 1000;
export const HF_HTTP_NOT_FOUND = 404;
export const HF_HTTP_SERVER_ERROR_MIN = 500;

// Search limits

export const HF_DEFAULT_LIMIT = 50;
export const HF_MAX_LIMIT = 100;

// GGUF shard files

/** Matches a split-shard GGUF file name, e.g. `Model-00001-of-00015.gguf`. */
export const HF_SHARD_REGEX = /-(\d{5})-of-(\d{5})\.gguf$/i;
/** Index (1-based) of the first shard in a split-shard set. */
export const HF_FIRST_SHARD = 1;
/** Zero-padded width of the shard index in a split-shard file name. */
export const HF_SHARD_PAD_WIDTH = 5;

// Quantization tokens

/** `UD-` (Unsloth Dynamic) custom quantization prefix, e.g. `UD-Q4_K_XL`. */
export const HF_UD_QUANT_PREFIX = 'UD';
export const HF_UD_QUANT_PREFIX_REGEX = /^UD-/i;
/**
 * Segment marking an Unsloth `shared-` draft head that borrows the target
 * model's embedding/output weights, e.g. `...-shared-Q4_K_M.gguf`.
 */
export const HF_SHARED_DRAFT_TOKEN = 'shared';
/**
 * Extracts the leading precision digits from a quant token, e.g.
 * `Q4_K_XL` -> 4, `IQ2_XXS` -> 2, `TQ1_0` -> 1, `BF16` -> 16.
 */
export const HF_QUANT_PRECISION_REGEX = /^(?:I?Q|TQ|BF|F|MXFP)?(\d+)/i;

// Model card tags

/** Matches the `base_model:` tag (plain or `quantized:`), capturing the repo id. */
export const HF_BASE_MODEL_TAG_REGEX = /^base_model:(?:quantized:)?(.+)$/;
export const HF_LICENSE_TAG_PREFIX = 'license:';
export const HF_GATED_TAG = 'gated';
export const HF_GGUF_TAG = 'gguf';
export const HF_SAFETENSORS_TAG = 'safetensors';

// Pipeline tasks (logic use only - matching `pipeline_tag` values against tags)

export const HF_TASK_TAGS: readonly string[] = [
	'audio-classification',
	'audio-to-audio',
	'automatic-speech-recognition',
	'conversational',
	'depth-estimation',
	'feature-extraction',
	'fill-mask',
	'image-classification',
	'image-feature-extraction',
	'image-segmentation',
	'image-text-to-text',
	'image-to-text',
	'image-to-video',
	'object-detection',
	'question-answering',
	'reinforcement-learning',
	'robotics',
	'sentence-similarity',
	'summarization',
	'text2text-generation',
	'text-classification',
	'text-generation',
	'text-to-image',
	'text-to-speech',
	'text-to-video',
	'token-classification',
	'translation',
	'video-to-video',
	'voice-activity-detection',
	'zero-shot-classification'
];

// Formatting

export const BYTE = 1;
export const KILOBYTE = 1_000;
export const MEGABYTE = 1_000_000;
export const GIGABYTE = 1_000_000_000;
export const TERABYTE = 1_000_000_000_000;

/**
 * Matches a human size string (`177GB`, `1.2 TB`, `500MB`), capturing the
 * numeric value and its unit suffix. Used by `parseSizeBytes`.
 */
// LLAMA-APP-REUSE: catalog size string parsing
export const HF_SIZE_STRING_REGEX = /^\s*([\d.]+)\s*([a-z]+)\s*$/i;

/**
 * Byte multiplier for a size suffix (`k` kilobyte, `m` megabyte, ...) as used by
 * the llama.app catalog `size` strings, whose suffix is lowercase.
 */
export const HF_SIZE_SUFFIX_BYTES: Readonly<Record<string, number>> = {
	b: BYTE,
	g: GIGABYTE,
	k: KILOBYTE,
	m: MEGABYTE,
	t: TERABYTE
};

export const BYTE_LABEL = 'B';
export const KILOBYTE_LABEL = 'KB';
export const MEGABYTE_LABEL = 'MB';
export const GIGABYTE_LABEL = 'GB';

/** Count suffixes for compact number formatting, e.g. `1.5K`, `2.0M`. */
export const KILO_LABEL = 'K';
export const MEGA_LABEL = 'M';
export const GIGA_LABEL = 'B';

// Relative time

export const MS_PER_DAY = 1000 * 60 * 60 * 24;
export const DAYS_PER_WEEK = 7;
/** Rough month length in days, used to bucket relative timestamps. */
export const DAYS_PER_MONTH = 30;
export const DAYS_PER_YEAR = 365;

export const TODAY_LABEL = 'Today';
export const YESTERDAY_LABEL = 'Yesterday';
export const DAYS_AGO_LABEL = 'days ago';
export const WEEKS_AGO_LABEL = 'weeks ago';
export const MONTHS_AGO_LABEL = 'months ago';
export const YEARS_AGO_LABEL = 'years ago';

// Cache paths

/**
 * Matches a local HF cache file path
 * (`.../models--<org>--<name>/snapshots/<sha>/<file>`), capturing the repo
 * directory name and the repo-relative file path.
 */
export const HF_CACHE_PATH_REGEX = /models--(.+?)\/snapshots\/[^/]+\/(.+)$/;
/** Separator between org and name segments in an HF cache directory name. */
export const HF_CACHE_DIR_SEPARATOR = '--';

// README

/** Matches a leading YAML frontmatter block (--- ... ---) in a markdown document. */
export const HF_FRONTMATTER_REGEX = /^---\r?\n[\s\S]*?\r?\n---\r?\n?/;

// Param counts

/**
 * Best-effort parameter count token in a model id/name, e.g. `27B` from
 * `Qwen3.8-27B-GGUF` or `300M` from `embeddinggemma-300M-GGUF`.
 */
export const HF_PARAM_COUNT_REGEX = /(?:^|[^a-z0-9])(\d+(?:[._]\d+)?)\s*([bm])(?![a-z0-9])/i;
