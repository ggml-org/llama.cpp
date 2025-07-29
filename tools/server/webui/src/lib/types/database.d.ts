export interface Conversation {
	id: string;
	lastModified: number;
	currNode: string;
	name: string;
}

export interface Message {
	id: string;
	convId: string;
	type: 'root' | 'text' | 'think';
	timestamp: number;
	role: 'system' | 'user' | 'assistant';
	content: string;
	parent: string;
	children: string[];
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
