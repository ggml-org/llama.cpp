export interface ApiChatMessageContentPart {
	type: 'text' | 'image_url';
	text?: string;
	image_url?: {
		url: string;
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
			typical_p: number;
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
			logit_bias: any[];
			n_probs: number;
			min_keep: number;
			grammar: string;
			grammar_lazy: boolean;
			grammar_triggers: any[];
			preserved_tokens: any[];
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
			lora: any[];
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
	temperature?: number;
	max_tokens?: number;
}

export interface ApiChatCompletionStreamChunk {
	choices: Array<{
		delta: {
			content?: string;
		};
	}>;
	timings?: {
		prompt_n?: number;
		prompt_ms?: number;
		predicted_n?: number;
		predicted_ms?: number;
	};
}

export interface ApiChatCompletionResponse {
	choices: Array<{
		message: {
			content: string;
		};
	}>;
}
