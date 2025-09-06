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
