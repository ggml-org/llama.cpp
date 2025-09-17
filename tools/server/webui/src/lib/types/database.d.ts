import type { ChatMessageTimings } from './chat';

export interface DatabaseConversation {
	currNode: string | null;
	id: string;
	lastModified: number;
	name: string;
}

export interface DatabaseMessageExtraAudioFile {
	type: 'audioFile';
	name: string;
	base64Data: string;
	mimeType: string;
}

export interface DatabaseMessageExtraImageFile {
	type: 'imageFile';
	name: string;
	base64Url: string;
}

export interface DatabaseMessageExtraTextFile {
	type: 'textFile';
	name: string;
	content: string;
}

export interface DatabaseMessageExtraPdfFile {
	type: 'pdfFile';
	name: string;
	content: string; // Text content extracted from PDF
	images?: string[]; // Optional: PDF pages as base64 images
	processedAsImages: boolean; // Whether PDF was processed as images
}

export type DatabaseMessageExtra =
	| DatabaseMessageExtraImageFile
	| DatabaseMessageExtraTextFile
	| DatabaseMessageExtraAudioFile
	| DatabaseMessageExtraPdfFile;

export interface DatabaseMessage {
	id: string;
	convId: string;
	type: ChatMessageType;
	timestamp: number;
	role: ChatRole;
	content: string;
	parent: string;
	thinking: string;
	children: string[];
	extra?: DatabaseMessageExtra[];
	timings?: ChatMessageTimings;
}
