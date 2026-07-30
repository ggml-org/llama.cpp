import type {
	ContentPartType,
	FileTypeAudio,
	ServerModelStatus,
	ServerModelsSseEventType,
	ServerRole
} from '$lib/enums';
import type { ChatMessagePromptProgress, ChatRole } from './chat';

export type AudioInputFormat = FileTypeAudio.WAV | FileTypeAudio.MP3;

export interface ApiChatCompletionToolFunction {
	name: string;
	description?: string;
	parameters: Record<string, unknown>;
}

export interface ApiChatCompletionTool {
	type: 'function';
	function: ApiChatCompletionToolFunction;
}

export interface ApiChatMessageContentPart {
	type: ContentPartType;
	text?: string;
	image_url?: {
		url: string;
	};
	input_audio?: {
		data: string;
		format: AudioInputFormat;
	};
	input_video?: {
		data: string;
		format: 'mp4' | 'ogg' | 'auto';
	};
}

export interface ApiContextSizeError {
	code: number;
	message: string;
	type: 'exceed_context_size_error';
	n_prompt_tokens: number;
	n_ctx: number;
}

export interface ApiErrorResponse {
	error:
		| ApiContextSizeError
		| {
				code: number;
				message: string;
				type?: string;
		  };
}

export interface ApiChatMessageData {
	role: ChatRole;
	content: string | ApiChatMessageContentPart[];
	reasoning_content?: string;
	tool_calls?: ApiChatCompletionToolCall[];
	tool_call_id?: string;
	timestamp?: number;
}

/**
 * Model status object from /models endpoint
 */
export interface ApiModelStatus {
	/** Status value: loaded, unloaded, loading, sleeping, failed */
	value: ServerModelStatus;
	/** Command line arguments used when loading (only for loaded models) */
	args?: string[];
}

/**
 * Model entry from /models endpoint (ROUTER mode)
 * Based on actual API response structure
 */
export interface ApiModelDataEntry {
	/** Model identifier (e.g., "ggml-org/Qwen2.5-Omni-7B-GGUF:latest") */
	id: string;
	/** Model name (optional, usually same as id - not always returned by API) */
	name?: string;
	/** Object type, always "model" */
	object: string;
	/** Owner, usually "llamacpp" */
	owned_by: string;
	/** Creation timestamp */
	created: number;
	/** Whether model files are in HuggingFace cache */
	in_cache: boolean;
	/** Path to model manifest file */
	path: string;
	/** Current status of the model */
	status: ApiModelStatus;
	/** Alternative names that resolve to this model */
	aliases?: string[];
	/** Informational tags for this model */
	tags?: string[];
	/** Legacy meta field (may be present in older responses) */
	meta?: Record<string, unknown> | null;
}

/**
 * Load stage reported by the /models/sse feed, in load order.
 */
export type ApiModelLoadStage = 'text_model' | 'spec_model' | 'mmproj_model';

/**
 * Load progress snapshot: the full ordered stage plan, the active stage,
 * and its fractional value (0.0 -> 1.0).
 */
export interface ApiModelsSseProgress {
	stages: ApiModelLoadStage[];
	current: ApiModelLoadStage;
	value: number;
}

/**
 * Status payload carried by a /models/sse envelope.
 * exit_code appears on unload.
 */
export interface ApiModelsSseData {
	status: ServerModelStatus;
	progress?: ApiModelsSseProgress;
	exit_code?: number;
}

/**
 * Event kind multiplexed on the /models/sse feed.
 * Only the status_* events carry a status payload, models_reload signals a
 * full list refresh, model_remove drops a row, download_* drive download UI.
 */
/**
 * One /models/sse record. event discriminates the kind, model names the
 * target instance, data carries the status payload when present.
 */
export interface ApiModelsSseEvent {
	model: string;
	event: ServerModelsSseEventType;
	data: ApiModelsSseData;
}

export interface ApiModelDetails {
	name: string;
	model: string;
	modified_at?: string;
	size?: string | number;
	digest?: string;
	type?: string;
	description?: string;
	tags?: string[];
	capabilities?: string[];
	parameters?: string;
	details?: {
		parent_model?: string;
		format?: string;
		family?: string;
		families?: string[];
		parameter_size?: string;
		quantization_level?: string;
	};
}

export interface ApiModelListResponse {
	object: string;
	data: ApiModelDataEntry[];
	models?: ApiModelDetails[];
}

export interface ApiLlamaCppServerProps {
	default_generation_settings: {
		id: number;
		id_task: number;
		n_ctx: number;
		speculative: boolean;
		is_processing: boolean;
		params: {
			n_predict: number;
			seed: number;
			temperature: number;
			dynatemp_range: number;
			dynatemp_exponent: number;
			top_k: number;
			top_p: number;
			min_p: number;
			top_n_sigma: number;
			xtc_probability: number;
			xtc_threshold: number;
			typ_p: number;
			repeat_last_n: number;
			repeat_penalty: number;
			presence_penalty: number;
			frequency_penalty: number;
			dry_multiplier: number;
			dry_base: number;
			dry_allowed_length: number;
			dry_penalty_last_n: number;
			dry_sequence_breakers: string[];
			mirostat: number;
			mirostat_tau: number;
			mirostat_eta: number;
			stop: string[];
			max_tokens: number;
			n_keep: number;
			n_discard: number;
			ignore_eos: boolean;
			stream: boolean;
			logit_bias: Array<[number, number]>;
			n_probs: number;
			min_keep: number;
			grammar: string;
			grammar_lazy: boolean;
			grammar_triggers: string[];
			preserved_tokens: number[];
			chat_format: string;
			reasoning_format: string;
			reasoning_in_content: boolean;
			generation_prompt: string;
			samplers: string[];
			backend_sampling: boolean;
			'speculative.n_max': number;
			'speculative.n_min': number;
			'speculative.p_min': number;
			timings_per_token: boolean;
			post_sampling_probs: boolean;
			lora: Array<{ name: string; scale: number }>;
		};
		prompt: string;
		next_token: {
			has_next_token: boolean;
			has_new_line: boolean;
			n_remain: number;
			n_decoded: number;
			stopping_word: string;
		};
	};
	total_slots: number;
	model_path: string;
	role: ServerRole;
	modalities: {
		vision: boolean;
		audio: boolean;
		video: boolean;
	};
	chat_template: string;
	bos_token: string;
	eos_token: string;
	build_info: string;
	/** @deprecated Use {@link ui_settings} instead */
	webui_settings?: Record<string, string | number | boolean>;
	ui_settings?: Record<string, string | number | boolean>;
	cors_proxy_enabled?: boolean;
	agent_mode?: boolean;
}

export interface ApiChatCompletionRequest {
	messages: Array<{
		role: ChatRole;
		content: string | ApiChatMessageContentPart[];
		reasoning_content?: string;
		tool_calls?: ApiChatCompletionToolCall[];
		tool_call_id?: string;
	}>;
	stream?: boolean;
	model?: string;
	return_progress?: boolean;
	sse_ping_interval?: number;
	tools?: ApiChatCompletionTool[];
	// Reasoning parameters
	reasoning_format?: string;
	// Generation parameters
	temperature?: number;
	max_tokens?: number;
	// Sampling parameters
	dynatemp_range?: number;
	dynatemp_exponent?: number;
	top_k?: number;
	top_p?: number;
	min_p?: number;
	xtc_probability?: number;
	xtc_threshold?: number;
	typ_p?: number;
	// Penalty parameters
	repeat_last_n?: number;
	repeat_penalty?: number;
	presence_penalty?: number;
	frequency_penalty?: number;
	dry_multiplier?: number;
	dry_base?: number;
	dry_allowed_length?: number;
	dry_penalty_last_n?: number;
	// Sampler configuration
	samplers?: string[];
	backend_sampling?: boolean;
	// Custom parameters (JSON string)
	custom?: Record<string, unknown>;
	timings_per_token?: boolean;
	// Continuation control (vLLM compat)
	add_generation_prompt?: boolean;
	continue_final_message?: boolean;
}

export interface ApiChatCompletionToolCallFunctionDelta {
	name?: string;
	arguments?: string;
}

export interface ApiChatCompletionToolCallDelta {
	index?: number;
	id?: string;
	type?: string;
	function?: ApiChatCompletionToolCallFunctionDelta;
}

export interface ApiChatCompletionToolCall extends ApiChatCompletionToolCallDelta {
	function?: ApiChatCompletionToolCallFunctionDelta & { arguments?: string };
}

export interface ApiChatCompletionStreamChunk {
	id?: string;
	object?: string;
	model?: string;
	choices: Array<{
		model?: string;
		metadata?: { model?: string };
		delta: {
			content?: string;
			reasoning_content?: string;
			model?: string;
			tool_calls?: ApiChatCompletionToolCallDelta[];
		};
		finish_reason?: string | null;
	}>;
	timings?: {
		prompt_n?: number;
		prompt_ms?: number;
		predicted_n?: number;
		predicted_ms?: number;
		cache_n?: number;
	};
	prompt_progress?: ChatMessagePromptProgress;
}

export interface ApiChatCompletionResponse {
	model?: string;
	choices: Array<{
		model?: string;
		metadata?: { model?: string };
		message: {
			content: string;
			reasoning_content?: string;
			model?: string;
			tool_calls?: ApiChatCompletionToolCall[];
		};
		finish_reason?: string | null;
	}>;
}

export interface ApiSlotData {
	id: number;
	id_task: number;
	n_ctx: number;
	speculative: boolean;
	is_processing: boolean;
	params: {
		n_predict: number;
		seed: number;
		temperature: number;
		dynatemp_range: number;
		dynatemp_exponent: number;
		top_k: number;
		top_p: number;
		min_p: number;
		top_n_sigma: number;
		xtc_probability: number;
		xtc_threshold: number;
		typical_p: number;
		repeat_last_n: number;
		repeat_penalty: number;
		presence_penalty: number;
		frequency_penalty: number;
		dry_multiplier: number;
		dry_base: number;
		dry_allowed_length: number;
		dry_penalty_last_n: number;
		mirostat: number;
		mirostat_tau: number;
		mirostat_eta: number;
		max_tokens: number;
		n_keep: number;
		n_discard: number;
		ignore_eos: boolean;
		stream: boolean;
		n_probs: number;
		min_keep: number;
		chat_format: string;
		reasoning_format: string;
		reasoning_in_content: boolean;
		generation_prompt: string;
		samplers: string[];
		backend_sampling: boolean;
		'speculative.n_max': number;
		'speculative.n_min': number;
		'speculative.p_min': number;
		timings_per_token: boolean;
		post_sampling_probs: boolean;
		lora: Array<{ name: string; scale: number }>;
	};
	next_token: {
		has_next_token: boolean;
		has_new_line: boolean;
		n_remain: number;
		n_decoded: number;
	};
}

export interface ApiProcessingState {
	status: 'initializing' | 'generating' | 'preparing' | 'idle';
	tokensDecoded: number;
	tokensRemaining: number;
	contextUsed: number;
	contextTotal: number | null;
	outputTokensUsed: number; // Total output tokens (thinking + regular content)
	outputTokensMax: number; // Max output tokens allowed
	temperature: number;
	topP: number;
	speculative: boolean;
	hasNextToken: boolean;
	tokensPerSecond?: number;
	// Progress information from prompt_progress
	progressPercent?: number;
	promptProgress?: ChatMessagePromptProgress;
	promptTokens?: number;
	promptMs?: number;
	cacheTokens?: number;
}

/**
 * Router model metadata - extended from ApiModelDataEntry with additional router-specific fields
 * @deprecated Use ApiModelDataEntry instead - the /models endpoint returns this structure directly
 */
export interface ApiRouterModelMeta {
	/** Model identifier (e.g., "ggml-org/Qwen2.5-Omni-7B-GGUF:latest") */
	name: string;
	/** Path to model file or manifest */
	path: string;
	/** Optional path to multimodal projector */
	path_mmproj?: string;
	/** Whether model is in HuggingFace cache */
	in_cache: boolean;
	/** Port where model instance is running (0 if not loaded) */
	port?: number;
	/** Current status of the model */
	status: ApiModelStatus;
	/** Error message if status is FAILED */
	error?: string;
}

/**
 * Request to load a model
 */
export interface ApiRouterModelsLoadRequest {
	model: string;
}

/**
 * Response from loading a model
 */
export interface ApiRouterModelsLoadResponse {
	success: boolean;
	error?: string;
}

/**
 * Request to check model status
 */
export interface ApiRouterModelsStatusRequest {
	model: string;
}

/**
 * Response with model status
 */
export interface ApiRouterModelsStatusResponse {
	model: string;
	status: ModelStatus;
	port?: number;
	error?: string;
}

/**
 * Response with list of all models from /models endpoint
 * Note: This is the same as ApiModelListResponse - the endpoint returns the same structure
 * regardless of server mode (MODEL or ROUTER)
 */
export interface ApiRouterModelsListResponse {
	object: string;
	data: ApiModelDataEntry[];
}

/**
 * Request to unload a model
 */
export interface ApiRouterModelsUnloadRequest {
	model: string;
}

/**
 * Response from unloading a model
 */
export interface ApiRouterModelsUnloadResponse {
	success: boolean;
	error?: string;
}

/**
 * Entry returned by POST /v1/streams/lookup. The client passes the conv ids it owns in the body
 * and the server returns one entry per matching live or recently completed background streaming
 * session, keyed by conversation_id. The WebUI uses this at mount and on visibilitychange to
 * populate sidebar spinners and to reattach to an ongoing inference for the active conversation.
 * The server never lists ids the client did not ask about, so foreign random UUIDs stay private.
 */
export interface ApiStreamSession {
	conversation_id: string;
	is_done: boolean;
	total_bytes: number;
	started_at: number;
	completed_at: number;
}

/**
 * One entry in the response of `POST /v1/filesystem/search`.
 * `path` and `parent` are canonical absolute paths on the server's filesystem;
 * `name` is the basename of the entry.
 * `size` is only populated for files; both kinds expose `modified` in unix
 * seconds since epoch.
 */
export interface ApiFilesystemSearchEntry {
	name: string;
	path: string;
	parent: string;
	type: 'file' | 'directory';
	size?: number;
	modified: number;
}

/**
 * Request body for `POST /v1/filesystem/search`. All fields except `query` are
 * optional. `path` defaults to the server process's current working directory
 * on the server side, so an empty value searches from there.
 */
export interface ApiFilesystemSearchRequest {
	query: string;
	path?: string;
	type?: 'any' | 'file' | 'directory';
	match?: 'substring' | 'prefix';
	limit?: number;
	max_depth?: number;
	/** When false (default) drops entries under dot-prefixed directories. */
	show_hidden?: boolean;
}

export interface ApiFilesystemSearchResponse {
	results: ApiFilesystemSearchEntry[];
}

/**
 * One browse root returned by `GET /v1/filesystem/roots`. Each root is a
 * canonical absolute directory the user is allowed to search from. Servers
 * fall back to `$HOME` when no `--browse-root` is configured.
 */
export interface ApiFilesystemRoot {
	/** Canonical absolute path - safe to send back as `path` in search calls. */
	path: string;
	/** True for the root the server uses as the implicit default for empty `path`. */
	default: boolean;
}

export interface ApiFilesystemRootsResponse {
	roots: ApiFilesystemRoot[];
}

/**
 * Request body for `POST /v1/filesystem/git`. The server resolves `path`
 * against the configured browse roots (mirroring `/filesystem/search`)
 * and walks upward looking for `.git/`. Pass an empty `path` to probe the
 * default browse root.
 */
export interface ApiFilesystemGitRequest {
	path?: string;
}

/**
 * Response from `POST /v1/filesystem/git`. `is_repo=false` is a valid
 * outcome (the path simply isn't inside a git repository) and is not
 * surfaced as an HTTP error - the UI just hides the branch badge.
 *
 * `branch` is populated for the common `ref: refs/heads/<name>` case,
 * the literal string `"detached"` when `.git/HEAD` contains a bare SHA,
 * and `"submodule"` when `.git` is a gitfile (we don't chase the linked
 * gitdir, so we can't resolve a branch in that layout).
 */
export interface ApiFilesystemGitResponse {
	/** Canonical absolute path the server actually probed. */
	path: string;
	/** True when `.git` was found somewhere on the way up. */
	is_repo: boolean;
	/** Repo root - the directory that holds `.git/`. Empty when not a repo. */
	root: string;
	/** Branch name, "detached", "submodule", or empty when not a repo. */
	branch: string;
}
