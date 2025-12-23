import type { ServerModelStatus, ServerRole } from '$lib/enums';
import type { ChatMessagePromptProgress } from './chat';

export interface ApiChatMessageContentPart {
	type: 'text' | 'image_url' | 'input_audio';
	text?: string;
	image_url?: {
		url: string;
	};
	input_audio?: {
		data: string;
		format: 'wav' | 'mp3';
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
	timestamp?: number;
	/**
	 * Optional reasoning/thinking content to be sent back to the server.
	 *
	 * llama-server accepts this non-OpenAI field and uses it to preserve the model's
	 * internal "thinking" blocks across tool-call resumptions (notably for gpt-oss).
	 */
	reasoning_content?: string;
	tool_call_id?: string;
	tool_calls?: ApiChatCompletionToolCallDelta[];
}

/**
 * Model status object from /models endpoint
 */
export interface ApiModelStatus {
	/** Status value: loaded, unloaded, loading, failed */
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
	/** Legacy meta field (may be present in older responses) */
	meta?: Record<string, unknown> | null;
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
			thinking_forced_open: boolean;
			samplers: string[];
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
	};
	chat_template: string;
	bos_token: string;
	eos_token: string;
	build_info: string;
	webui_settings?: Record<string, string | number | boolean>;
}

export interface ApiChatCompletionRequestMessage {
	role: ChatRole;
	content: string | ApiChatMessageContentPart[];
	reasoning_content?: string;
	tool_call_id?: string;
	tool_calls?: ApiChatCompletionToolCallDelta[];
}

export interface ApiChatCompletionRequest {
	messages: Array<ApiChatCompletionRequestMessage>;
	stream?: boolean;
	model?: string;
	tools?: ApiToolDefinition[];
	tool_choice?: 'auto' | 'none' | { type: 'function'; function: { name: string } };
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
	// Custom parameters (JSON string)
	custom?: Record<string, unknown>;
	timings_per_token?: boolean;
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

export interface ApiToolDefinition {
	type: 'function';
	function: {
		name: string;
		description?: string;
		parameters: Record<string, unknown>;
	};
}

export interface ApiChatCompletionStreamChunk {
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
			tool_calls?: ApiChatCompletionToolCallDelta[];
		};
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
		thinking_forced_open: boolean;
		samplers: string[];
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
	contextTotal: number;
	outputTokensUsed: number; // Total output tokens (thinking + regular content)
	outputTokensMax: number; // Max output tokens allowed
	temperature: number;
	topP: number;
	speculative: boolean;
	hasNextToken: boolean;
	tokensPerSecond?: number;
	// Progress information from prompt_progress
	progressPercent?: number;
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
