/**
 * HuggingFace Hub enums.
 *
 * Values mirror the strings used by the HF REST API
 * (https://huggingface.co/docs/huggingface_hub/package_reference/hf_api)
 * so they can be sent and compared directly.
 */
// LLAMA-APP-REUSE: HF sort and sidecar-form enums

/** Sort field for /api/models search queries. */
export enum HfModelSort {
	CREATED_AT = 'createdAt',
	DOWNLOADS = 'downloads',
	LAST_MODIFIED = 'lastModified',
	LIKES = 'likes',
	TRENDING_SCORE = 'trendingScore'
}

/**
 * Where the sidecar token (`mtp` / `dflash` / `mmproj` / ...) sits in the
 * filename.
 * - `prefix`  sidecar file that lives next to the main weights, e.g. `mtp-Q4_0.gguf`
 * - `suffix`  embedded draft baked into the main weights, e.g. `Hy3-IQ1_M-mtp.gguf`
 */
export enum SidecarForm {
	PREFIX = 'prefix',
	SUFFIX = 'suffix'
}

/** Entry type in a model repository file tree (`/tree` responses). */
export enum HfEntryType {
	DIRECTORY = 'directory',
	FILE = 'file'
}
