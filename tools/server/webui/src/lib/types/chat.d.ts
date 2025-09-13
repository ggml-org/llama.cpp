export type ChatMessageType = 'root' | 'text' | 'think';
export type ChatRole = 'user' | 'assistant' | 'system';

export interface ChatUploadedFile {
	id: string;
	name: string;
	size: number;
	type: string;
	file: File;
	preview?: string;
	textContent?: string;
}

export interface MessageSiblingInfo {
	message: DatabaseMessage;
	siblingIds: string[];
	currentIndex: number;
	totalSiblings: number;
}

export interface MessagePromptProgress {
	cache: number;
	processed: number;
	time_ms: number;
	total: number;
}

export interface MessageTimings {
	cache_n?: number;
	predicted_ms?: number;
	predicted_n?: number;
	prompt_ms?: number;
	prompt_n?: number;
}
