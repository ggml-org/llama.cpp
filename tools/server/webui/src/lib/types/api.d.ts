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
	modalities: {
		vision: boolean;
		audio: boolean;
	};
	chat_template: string;
	bos_token: string;
	eos_token: string;
	build_info: string;
}

export interface ApiChatCompletionRequest {
	messages: Array<{
		role: ChatRole;
		content: string | ApiChatMessageContentPart[];
	}>;
	stream?: boolean;
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
}

export interface ApiChatCompletionStreamChunk {
	choices: Array<{
		delta: {
			content?: string;
			reasoning_content?: string;
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
	choices: Array<{
		message: {
			content: string;
			reasoning_content?: string;
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
	cacheTokens?: number;
}
