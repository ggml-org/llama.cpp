import type { ChatMessageData } from './chat';

export interface DatabaseChatMessage extends ChatMessageData {
	id: string;
	chatId: string;
	timestamp: number;
	tokens?: number;
	model?: string;
}

export interface DatabaseChat {
	id: string;
	name: string;
	createdAt: number;
	updatedAt: number;
	messageCount: number;
	model?: string;
	systemPrompt?: string;
}

export interface DatabaseAppSettings {
	id: string;
	theme: 'light' | 'dark' | 'system';
	model: string;
	temperature: number;
	maxTokens: number;
	topP: number;
	topK: number;
	repeatPenalty: number;
	seed: number;
	systemPrompt: string;
}
