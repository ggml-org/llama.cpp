import { PATH_SEPARATOR } from '$lib/constants';
import {
	BYTE,
	BYTE_LABEL,
	DAYS_AGO_LABEL,
	DAYS_PER_MONTH,
	DAYS_PER_WEEK,
	DAYS_PER_YEAR,
	GIGA_LABEL,
	GIGABYTE,
	GIGABYTE_LABEL,
	HF_API_MODELS_URL,
	HF_AVATARS_URL,
	HF_BASE_MODEL_TAG_REGEX,
	HF_BASE_URL,
	HF_CACHE_DIR_SEPARATOR,
	HF_CACHE_PATH_REGEX,
	HF_DEFAULT_LIMIT,
	HF_FIRST_SHARD,
	HF_FRONTMATTER_REGEX,
	HF_FULL_DETAIL_PARAM,
	HF_GATED_TAG,
	HF_GGUF_FILTER,
	HF_GGUF_TAG,
	HF_HTTP_NOT_FOUND,
	HF_HTTP_SERVER_ERROR_MIN,
	HF_LICENSE_TAG_PREFIX,
	HF_LINK_HEADER,
	HF_LINK_NEXT_REGEX,
	HF_MAIN_BRANCH,
	HF_MAX_LIMIT,
	HF_MODEL_LIST_EXPAND,
	HF_PARAM_COUNT_REGEX,
	HF_QUANT_PRECISION_REGEX,
	HF_RAW_PATH,
	HF_README_FILENAME,
	HF_RECURSIVE_TREE_PARAM,
	HF_RETRY_ATTEMPTS,
	HF_RETRY_DELAY_MS,
	HF_SAFETENSORS_TAG,
	HF_SHARD_PAD_WIDTH,
	HF_SHARD_REGEX,
	HF_SHARED_DRAFT_TOKEN,
	HF_SIZE_STRING_REGEX,
	HF_SIZE_SUFFIX_BYTES,
	HF_TASK_TAGS,
	HF_TREE_PATH,
	HF_UD_QUANT_PREFIX,
	HF_UD_QUANT_PREFIX_REGEX,
	KILO_LABEL,
	KILOBYTE,
	KILOBYTE_LABEL,
	MEGA_LABEL,
	MEGABYTE,
	MEGABYTE_LABEL,
	MODELS_DISCOVER_CATALOG_URL,
	MONTHS_AGO_LABEL,
	MS_PER_DAY,
	TODAY_LABEL,
	WEEKS_AGO_LABEL,
	YEARS_AGO_LABEL,
	YESTERDAY_LABEL
} from '$lib/constants';
import { MODEL_ID, type ModelSidecar, sidecarFromFileToken } from '$lib/constants';
import { HfEntryType, HfModelSort, SidecarForm } from '$lib/enums';
import type {
	HfCatalogEntry,
	HfModelDetailInfo,
	HfModelInfo,
	HfModelSearchParams,
	HfModelSibling
} from '$lib/types/huggingface';

/**
 * HuggingFaceService - Service for browsing and searching GGUF models on Hugging Face Hub
 */
export class HuggingFaceService {
	private static readonly BASE_URL = HF_API_MODELS_URL;

	// Cached base model lookups keyed by repo id, so repeated selector opens
	// never re-hit the HF API for the same repo.
	private static baseModelCache = new Map<string, { org: string; name: string } | null>();

	private static baseModelPending = new Map<
		string,
		Promise<{ org: string; name: string } | null>
	>();

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
	 * Collapse split GGUF shard sets (`-00001-of-00015.gguf`, ...) to their first
	 * shard, summing every shard's size so the kept entry reflects the whole
	 * quant. Non-sharded files pass through unchanged. Downloads are tag-based
	 * (`repo:quant`), so the first shard is enough to represent the set.
	 */
	static collapseGgufShards(siblings: HfModelSibling[]): HfModelSibling[] {
		const sizeByPath = new Map(siblings.map((f) => [f.path, f.size ?? 0]));
		const result: HfModelSibling[] = [];

		for (const file of siblings) {
			const match = HF_SHARD_REGEX.exec(file.path);

			if (!match) {
				result.push(file);

				continue;
			}

			// Keep only the first shard; its size becomes the whole shard set's.
			if (Number(match[1]) !== HF_FIRST_SHARD) continue;

			const total = Number(match[2]);
			const stem = file.path.slice(0, file.path.length - match[0].length);

			let size = 0;

			for (let i = HF_FIRST_SHARD; i <= total; i++) {
				const shard = HuggingFaceService.shardPath(stem, i, total);

				size += sizeByPath.get(shard) ?? 0;
			}

			result.push({ ...file, size });
		}

		return result;
	}

	// GGUF Model Browsing

	/**
	 * Extract the GGUF quantization token (e.g. `Q4_K_M`) and any sidecar type
	 * (`mtp`, `dflash`, `mmproj`, ...) from a `.gguf` filename. The sidecar token
	 * shows up either as a sidecar prefix (`mtp-<name>.gguf`, `dflash-<name>.gguf`,
	 * `mmproj-<name>.gguf`), as a `-mtp` suffix, or as the whole filename
	 * (`imatrix.gguf`); a `-draft` tail marks a standalone sidecar file
	 * (`Model-MTP-draft.gguf`).
	 *
	 * `sidecarForm` records which side of the filename the sidecar token sat
	 * on so callers can render badges differently (e.g. prefix on the left of
	 * the quant label, suffix appended to it).
	 * `quant` is `null` for files that don't carry a bit-depth token
	 * (e.g. `*-BF16.gguf`); `sidecar` is `null` if no sidecar flag is present.
	 * Returns `null` only when the filename doesn't end in `.gguf`.
	 */
	static extractQuantMeta(filename: string): {
		quant: string | null;
		/** Draft-head-only variant borrowing embed/output weights from the target model. */
		shared: boolean;
		sidecar: ModelSidecar | null;
		sidecarForm: SidecarForm | null;
	} | null {
		if (!MODEL_ID.WEIGHT_EXTENSION_REGEX.test(filename)) return null;

		// HF repos may nest sidecars in a folder (e.g. `MTP/mtp-Model-Q4_0.gguf`);
		// parse the file name only, the folder adds no quant information.
		let source = (filename.split(PATH_SEPARATOR).pop() ?? filename).replace(
			MODEL_ID.WEIGHT_EXTENSION_REGEX,
			''
		);
		let sidecar: ModelSidecar | null = null;
		let sidecarForm: SidecarForm | null = null;

		// A file named just the sidecar token (`imatrix.gguf`) is the sidecar
		// itself: no name or quant segments to parse.
		const bareSidecar = sidecarFromFileToken(source.toLowerCase());

		if (bareSidecar) {
			return { quant: null, shared: false, sidecar: bareSidecar, sidecarForm: SidecarForm.PREFIX };
		}

		const prefixMatch = source.match(MODEL_ID.SIDECAR_PREFIX_REGEX);

		if (prefixMatch) {
			sidecar = sidecarFromFileToken(prefixMatch[1].toLowerCase());
			sidecarForm = SidecarForm.PREFIX;
			source = prefixMatch[2];
		} else {
			const suffixMatch = source.match(MODEL_ID.SIDECAR_SUFFIX_REGEX);

			if (suffixMatch) {
				// Take the suffix sidecar even when the head carries no quant:
				// embedded drafts end in one (`Hy3-IQ1_M-mtp`), standalone sidecar
				// files do not (`Model-MTP-draft`, `Model-imatrix`).
				sidecar = sidecarFromFileToken(suffixMatch[2].toLowerCase());
				sidecarForm = SidecarForm.SUFFIX;
				source = suffixMatch[1];
			}
		}

		// Scan dash-separated segments left-to-right for the first quant match.
		// - For sidecars like `mtp-Q4_0-180MB.gguf` the quant is `Q4_0`.
		// - For embedded MTP like `Hy3-IQ1_M-mtp.gguf` we have `Hy3-IQ1_M` and `IQ1_M` matches.
		// - For main files like `Llama-3-8B-Q4_K_M.gguf` we land on the trailing quant.
		const segments = source.split(MODEL_ID.SEGMENT_SEPARATOR);
		const quantIdx = segments.findIndex((seg) => MODEL_ID.QUANTIZATION_SEGMENT_REGEX.test(seg));
		// Unsloth ships draft heads in two layouts: `shared-` files borrow the
		// embedding/output weights from the target model, others are self-contained.
		const shared = segments.some((seg) => seg.toLowerCase() === HF_SHARED_DRAFT_TOKEN);

		let quant = quantIdx >= 0 ? segments[quantIdx].toUpperCase() : null;

		// Recombine a `UD-` (Unsloth Dynamic) prefix, e.g. `...-UD-Q4_K_XL.gguf`.
		// The prefix must be the whole previous segment, matching the server's
		// `UD-<quant>` custom-quant convention (e.g. not `-mtp-Q4_K_M`).
		const udPrefixIdx = quantIdx - 1;

		if (quant && quantIdx > 0 && segments[udPrefixIdx].toUpperCase() === HF_UD_QUANT_PREFIX) {
			quant = `${HF_UD_QUANT_PREFIX}-${quant}`;
		}

		return { quant, shared, sidecar, sidecarForm };
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
	 * Format model downloads count with K/M/B suffix
	 */
	static formatDownloads(downloads: number): string {
		if (downloads >= GIGABYTE) {
			return `${(downloads / GIGABYTE).toFixed(1)}${GIGA_LABEL}`;
		}

		if (downloads >= MEGABYTE) {
			return `${(downloads / MEGABYTE).toFixed(1)}${MEGA_LABEL}`;
		}

		if (downloads >= KILOBYTE) {
			return `${(downloads / KILOBYTE).toFixed(1)}${KILO_LABEL}`;
		}

		return downloads.toString();
	}

	/**
	 * Format file size in bytes to human-readable string
	 */
	static formatFileSize(bytes: number): string {
		if (bytes >= GIGABYTE) {
			return `${(bytes / GIGABYTE).toFixed(1)} ${GIGABYTE_LABEL}`;
		}

		if (bytes >= MEGABYTE) {
			return `${(bytes / MEGABYTE).toFixed(1)} ${MEGABYTE_LABEL}`;
		}

		if (bytes >= KILOBYTE) {
			return `${(bytes / KILOBYTE).toFixed(1)} ${KILOBYTE_LABEL}`;
		}

		return `${bytes} ${BYTE_LABEL}`;
	}

	/**
	 * Format likes count with K suffix if applicable
	 */
	static formatLikes(likes: number): string {
		if (likes >= KILOBYTE) {
			return `${(likes / KILOBYTE).toFixed(1)}${KILO_LABEL}`;
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
		// timestamps can lie in the future (clock skew); clamp so they read as today
		const diffDays = Math.max(0, Math.floor(diffMs / MS_PER_DAY));

		if (diffDays === 0) return TODAY_LABEL;

		if (diffDays === 1) return YESTERDAY_LABEL;

		if (diffDays < DAYS_PER_WEEK) return `${diffDays} ${DAYS_AGO_LABEL}`;

		if (diffDays < DAYS_PER_MONTH) {
			return `${Math.floor(diffDays / DAYS_PER_WEEK)} ${WEEKS_AGO_LABEL}`;
		}

		if (diffDays < DAYS_PER_YEAR) {
			return `${Math.floor(diffDays / DAYS_PER_MONTH)} ${MONTHS_AGO_LABEL}`;
		}

		return `${Math.floor(diffDays / DAYS_PER_YEAR)} ${YEARS_AGO_LABEL}`;
	}

	/**
	 * Format a min-max size range with a single shared unit and no spaces
	 * around the dash, e.g. `19.0-28.6 GB`.
	 */
	static formatSizeRange(min: number, max: number): string {
		const unit =
			max >= GIGABYTE
				? GIGABYTE_LABEL
				: max >= MEGABYTE
					? MEGABYTE_LABEL
					: max >= KILOBYTE
						? KILOBYTE_LABEL
						: BYTE_LABEL;
		const div =
			unit === GIGABYTE_LABEL
				? GIGABYTE
				: unit === MEGABYTE_LABEL
					? MEGABYTE
					: unit === KILOBYTE_LABEL
						? KILOBYTE
						: BYTE;
		const fmt = (n: number) => (div === BYTE ? `${n}` : `${(n / div).toFixed(1)}`);

		return `${fmt(min)}-${fmt(max)} ${unit}`;
	}

	// Model Details & Files

	/**
	 * Avatar URL for an author (org or user). 404s when the author does not
	 * exist, so callers should provide a fallback.
	 */
	static getAvatarUrl(author: string): string {
		return `${HF_AVATARS_URL}${PATH_SEPARATOR}${author}`;
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

			const [org, ...rest] = base.split(PATH_SEPARATOR);

			return { name: rest.join(PATH_SEPARATOR), org };
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
			.map((t) => HF_BASE_MODEL_TAG_REGEX.exec(t)?.[1])
			.filter((v): v is string => Boolean(v));

		return Array.from(new Set([...fromCard, ...fromTags]));
	}

	/**
	 * Look up the average bit-depth for a known GGUF quantization.
	 * Returns `null` for unrecognized tokens.
	 */
	static getBitDepth(quant: string): number | null {
		// Strip a leading `UD-` (Unsloth Dynamic) prefix before lookup.
		const base = quant.replace(HF_UD_QUANT_PREFIX_REGEX, '');
		const direct = HuggingFaceService.QUANT_BIT_DEPTH[base];

		if (direct !== undefined) return direct;

		// Fall back to the leading precision digits for variants missing from the
		// map, e.g. `Q4_K_XL` -> 4, `IQ2_XXS` -> 2, `TQ1_0` -> 1, `BF16` -> 16.
		const match = HF_QUANT_PRECISION_REGEX.exec(base);

		return match ? parseInt(match[1], 10) : null;
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
	 * Fetch the llama.app model catalog. Returns an empty array on failure so
	 * callers can fall back gracefully.
	 */
	static async getCatalog(): Promise<HfCatalogEntry[]> {
		try {
			const response = await fetch(MODELS_DISCOVER_CATALOG_URL);

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
		const url = `${HF_API_MODELS_URL}${PATH_SEPARATOR}${modelId}?${HF_FULL_DETAIL_PARAM}`;

		try {
			const response = await fetch(url);

			if (response.status === HF_HTTP_NOT_FOUND) return null;

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
		return `${HF_BASE_URL}${PATH_SEPARATOR}${modelId}`;
	}

	// Utility Methods

	/**
	 * Get most liked GGUF models
	 */
	static async getMostLiked(limit: number = HF_DEFAULT_LIMIT): Promise<HfModelInfo[]> {
		return this.search({ limit, sort: HfModelSort.LIKES });
	}

	/**
	 * Get newly released GGUF models
	 */
	static async getNew(limit: number = HF_DEFAULT_LIMIT): Promise<HfModelInfo[]> {
		return this.search({ limit, sort: HfModelSort.CREATED_AT });
	}

	/**
	 * Get most popular GGUF models by downloads
	 */
	static async getPopular(limit: number = HF_DEFAULT_LIMIT): Promise<HfModelInfo[]> {
		return this.search({ limit, sort: HfModelSort.DOWNLOADS });
	}

	/**
	 * Fetch the raw README.md for a repo, with the YAML frontmatter stripped.
	 */
	static async getReadme(modelId: string): Promise<string | null> {
		// Do not encode the modelId, it contains slashes for author/name
		const url = `${HF_BASE_URL}${PATH_SEPARATOR}${modelId}${PATH_SEPARATOR}${HF_RAW_PATH}${PATH_SEPARATOR}${HF_MAIN_BRANCH}${PATH_SEPARATOR}${HF_README_FILENAME}`;

		try {
			const response = await fetch(url);

			if (response.status === HF_HTTP_NOT_FOUND) return null;

			if (!response.ok) throw new Error(`Failed to fetch README: ${response.status}`);

			return HuggingFaceService.stripFrontmatter(await response.text());
		} catch (error) {
			console.error(`Error fetching README for ${modelId}:`, error);

			return null;
		}
	}

	/**
	 * Get repository file tree to list available GGUF variants. Recursive so
	 * repos that keep quants in per-quant subdirectories (e.g. `UD-Q4_K_XL/`)
	 * are included; follows cursor pagination for repos over one page.
	 */
	static async getTree(modelId: string): Promise<HfModelSibling[]> {
		const files: HfModelSibling[] = [];
		const firstUrl =
			`${HF_API_MODELS_URL}${PATH_SEPARATOR}${modelId}${PATH_SEPARATOR}${HF_TREE_PATH}` +
			`${PATH_SEPARATOR}${HF_MAIN_BRANCH}?${HF_RECURSIVE_TREE_PARAM}`;

		let url: string | null = firstUrl;

		try {
			while (url) {
				const response: Response = await fetch(url);

				if (!response.ok) return files;

				const data = (await response.json()) as HfModelSibling[];

				files.push(...data.filter((f) => f.type !== HfEntryType.DIRECTORY));

				url = HuggingFaceService.parseNextPageUrl(response.headers.get(HF_LINK_HEADER));
			}
		} catch {
			// Return whatever was fetched before the failure.
		}

		return files;
	}

	/**
	 * Get trending GGUF models
	 */
	static async getTrending(limit: number = HF_DEFAULT_LIMIT): Promise<HfModelInfo[]> {
		return this.search({ limit, sort: HfModelSort.TRENDING_SCORE });
	}

	/**
	 * Parse a local HF cache file path
	 * (`.../models--<org>--<name>/snapshots/<sha>/<file>`) into its repo id and
	 * repo-relative file path. Returns null when the path is not an HF cache path.
	 */
	static parseCachePath(path: string): { repo: string; file: string } | null {
		const match = HF_CACHE_PATH_REGEX.exec(path);

		if (!match) return null;

		const parts = match[1].split(HF_CACHE_DIR_SEPARATOR);

		if (parts.length < 2) return null;

		return {
			file: match[2],
			repo: `${parts[0]}${PATH_SEPARATOR}${parts.slice(1).join(HF_CACHE_DIR_SEPARATOR)}`
		};
	}

	/**
	 * Best-effort parameter count parsed from a model id/name, e.g. `27B` from
	 * `Qwen3.8-27B-GGUF` or `300M` from `embeddinggemma-300M-GGUF`. Returns null
	 * when no size token is present.
	 */
	static parseParamCount(name: string): string | null {
		const match = HF_PARAM_COUNT_REGEX.exec(name);

		if (!match) return null;

		return `${match[1]}${match[2].toUpperCase()}`;
	}

	/**
	 * Parse a human size string (`177GB`, `1.2 TB`, `500MB`) to bytes. Returns
	 * null when it carries no number or no known suffix, so callers can fall
	 * back to another source instead of showing a wrong size.
	 */
	static parseSizeBytes(size: string): number | null {
		const match = HF_SIZE_STRING_REGEX.exec(size);

		if (!match) return null;

		const value = parseFloat(match[1]);
		const multiplier = HF_SIZE_SUFFIX_BYTES[match[2].toLowerCase()];

		if (!Number.isFinite(value) || multiplier === undefined) return null;

		return value * multiplier;
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
		const license =
			tags
				.find((tag) => tag.startsWith(HF_LICENSE_TAG_PREFIX))
				?.replace(HF_LICENSE_TAG_PREFIX, '') || null;
		const isGated = tags.includes(HF_GATED_TAG);
		const isGguf = tags.includes(HF_GGUF_TAG);
		const isSafetensors = tags.includes(HF_SAFETENSORS_TAG);
		const tasks = tags.filter((tag) => HF_TASK_TAGS.includes(tag));

		return { isGated, isGguf, isSafetensors, license, tasks };
	}

	/**
	 * Search GGUF models with various filters and options.
	 *
	 * Always expands the fields the discover rows render (chat template, context
	 * length, siblings, ...) so a search result carries the same badges as a
	 * catalog entry; caller-provided `expand` entries are merged in.
	 */
	static async search(params: HfModelSearchParams = {}): Promise<HfModelInfo[]> {
		const { expand, limit = HF_DEFAULT_LIMIT, ...restParams } = params;
		const url = this.buildUrl({
			...restParams,
			expand: [...new Set([...HF_MODEL_LIST_EXPAND, ...(expand ?? [])])],
			filter: HF_GGUF_FILTER,
			limit: Math.min(limit, HF_MAX_LIMIT)
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
		try {
			const response = await fetch(url);

			if (!response.ok) {
				if (response.status === HF_HTTP_NOT_FOUND) {
					return [];
				}

				if (response.status >= HF_HTTP_SERVER_ERROR_MIN && attempt < HF_RETRY_ATTEMPTS) {
					await this.delay(HF_RETRY_DELAY_MS * attempt);

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
			if (attempt < HF_RETRY_ATTEMPTS) {
				await this.delay(HF_RETRY_DELAY_MS * attempt);

				return this.fetchWithRetry(url, attempt + 1);
			}

			throw error;
		}
	}

	// Internal Methods

	/** Extract the `rel="next"` URL from an RFC 5988 `Link` header, if present. */
	private static parseNextPageUrl(linkHeader: string | null): string | null {
		if (!linkHeader) return null;

		const match = HF_LINK_NEXT_REGEX.exec(linkHeader);

		return match ? match[1] : null;
	}

	/** Full path of one shard in a split-shard GGUF set. */
	private static shardPath(stem: string, index: number, total: number): string {
		const pad = (n: number) => String(n).padStart(HF_SHARD_PAD_WIDTH, '0');

		return `${stem}-${pad(index)}-of-${pad(total)}.gguf`;
	}

	/** Strip a leading YAML frontmatter block (--- ... ---) from a markdown document. */
	private static stripFrontmatter(text: string): string {
		const match = text.match(HF_FRONTMATTER_REGEX);

		return match ? text.slice(match[0].length) : text;
	}
}
