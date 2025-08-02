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
	build_info: string;
	model_path: string;
	n_ctx: number;
	modalities?: {
		vision: boolean;
		audio: boolean;
	};
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
