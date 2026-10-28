import photoUrl from '../../fixtures/assets/1.jpg';
import clipVideoUrl from '../../fixtures/assets/clip.webm';
import hfLogoUrl from '../../fixtures/assets/hf-logo.svg';
import toneAudioUrl from '../../fixtures/assets/tone.wav';
import { AttachmentType, SpecialFileType } from '$lib/enums';
import type {
	DatabaseMessageExtraAudioFile,
	DatabaseMessageExtraImageFile,
	DatabaseMessageExtraLegacyContext,
	DatabaseMessageExtraMcpPrompt,
	DatabaseMessageExtraMcpResource,
	DatabaseMessageExtraPdfFile,
	DatabaseMessageExtraTextFile,
	DatabaseMessageExtraVideoFile,
	MCPResourceAttachment
} from '$lib/types';

export { clipVideoUrl, hfLogoUrl, photoUrl, toneAudioUrl };

export const gradientImageUrl = `data:image/svg+xml;utf8,${encodeURIComponent(
	'<svg xmlns="http://www.w3.org/2000/svg" width="320" height="200"><defs><linearGradient id="g" x1="0" y1="0" x2="1" y2="1"><stop offset="0" stop-color="#f97316"/><stop offset="1" stop-color="#8b5cf6"/></linearGradient></defs><rect width="320" height="200" fill="url(#g)"/></svg>'
)}`;

export const sampleCode = `import { serverStore } from '$lib/stores';

export async function loadProps(): Promise<void> {
	await serverStore.fetch();

	if (serverStore.error) {
		console.error('Server unavailable:', serverStore.error);
	}
}
`;

export const sampleNotes = `Meeting notes - release planning

- Ship the matte glass surfaces behind a flag first.
- Collect feedback from the nightly build users.
- Remove the flag once the dark theme contrast is fixed.
- Follow up on the attachment preview performance numbers from last week.`;

export const samplePdfText = `Attention Is All You Need

Abstract. The dominant sequence transduction models are based on complex recurrent or convolutional neural networks that include an encoder and a decoder.`;

// Stored message attachments (DatabaseMessageExtra)

export const imageAttachment: DatabaseMessageExtraImageFile = {
	base64Url: photoUrl,
	name: 'mountain-lake.jpg',
	size: 44_891,
	type: AttachmentType.IMAGE
};

export const logoAttachment: DatabaseMessageExtraImageFile = {
	base64Url: hfLogoUrl,
	name: 'hf-logo.svg',
	size: 35_285,
	type: AttachmentType.IMAGE
};

export const gradientAttachment: DatabaseMessageExtraImageFile = {
	base64Url: gradientImageUrl,
	name: 'gradient.svg',
	size: 412,
	type: AttachmentType.IMAGE
};

export const textAttachment: DatabaseMessageExtraTextFile = {
	content: sampleCode,
	name: 'load-props.ts',
	size: sampleCode.length,
	type: AttachmentType.TEXT
};

export const notesAttachment: DatabaseMessageExtraTextFile = {
	content: sampleNotes,
	name: 'notes.md',
	size: sampleNotes.length,
	type: AttachmentType.TEXT
};

export const legacyContextAttachment: DatabaseMessageExtraLegacyContext = {
	content: 'Pasted context from the old UI.',
	name: 'Pasted content',
	type: AttachmentType.LEGACY_CONTEXT
};

// PDF pages are pre-rendered so the preview does not need pdf.js
export const pdfAttachment: DatabaseMessageExtraPdfFile = {
	base64Data: '',
	content: samplePdfText,
	images: [photoUrl, gradientImageUrl],
	name: 'attention-is-all-you-need.pdf',
	processedAsImages: false,
	size: 2_215_244,
	type: AttachmentType.PDF
};

export const pdfAsImagesAttachment: DatabaseMessageExtraPdfFile = {
	...pdfAttachment,
	name: 'scanned-invoice.pdf',
	processedAsImages: true,
	size: 351_048
};

export const audioAttachment: DatabaseMessageExtraAudioFile = {
	base64Data: '',
	mimeType: 'audio/wav',
	name: 'voice-memo.wav',
	size: 8_078,
	type: AttachmentType.AUDIO
};

export const videoAttachment: DatabaseMessageExtraVideoFile = {
	base64Data: '',
	mimeType: 'video/webm',
	name: 'screen-recording.webm',
	size: 8_101,
	type: AttachmentType.VIDEO
};

export const mcpPromptAttachment: DatabaseMessageExtraMcpPrompt = {
	arguments: { language: 'TypeScript', topic: 'error handling' },
	content:
		'Review the following TypeScript code and suggest improvements for error handling. Focus on edge cases.',
	name: 'code-review',
	promptName: 'code-review',
	serverName: 'github',
	type: AttachmentType.MCP_PROMPT
};

export const mcpResourceAttachment: DatabaseMessageExtraMcpResource = {
	content: '# llama.cpp\n\nInference of LLMs in pure C/C++.',
	mimeType: 'text/markdown',
	name: 'README.md',
	serverName: 'filesystem',
	type: AttachmentType.MCP_RESOURCE,
	uri: 'file:///workspace/llama.cpp/README.md'
};

export const mcpResourceImageAttachment: DatabaseMessageExtraMcpResource = {
	content: '',
	mimeType: 'image/png',
	name: 'architecture.png',
	serverName: 'filesystem',
	type: AttachmentType.MCP_RESOURCE,
	uri: 'file:///workspace/docs/architecture.png'
};

export function toMcpResourceAttachment(
	extra: DatabaseMessageExtraMcpResource,
	overrides: Partial<MCPResourceAttachment> = {}
): MCPResourceAttachment {
	return {
		id: `resource-${extra.name}`,
		resource: {
			mimeType: extra.mimeType,
			name: extra.name,
			serverName: extra.serverName,
			title: extra.name,
			uri: extra.uri
		},
		...overrides
	};
}

// Pending uploads (ChatUploadedFile)

export function createUploadedFile(
	name: string,
	type: string,
	extra: Partial<ChatUploadedFile> = {}
): ChatUploadedFile {
	const content = extra.textContent ?? '';

	return {
		file: new File([content], name, { type }),
		id: `upload-${name}`,
		name,
		size: extra.size ?? content.length,
		type,
		...extra
	};
}

export const uploadedImage = createUploadedFile('mountain-lake.jpg', 'image/jpeg', {
	preview: photoUrl,
	size: 44_891
});

export const uploadedLogo = createUploadedFile('hf-logo.svg', 'image/svg+xml', {
	preview: hfLogoUrl,
	size: 35_285
});

export const uploadedGradient = createUploadedFile('gradient.svg', 'image/svg+xml', {
	preview: gradientImageUrl,
	size: 412
});

export const uploadedText = createUploadedFile('load-props.ts', 'text/typescript', {
	textContent: sampleCode
});

export const uploadedNotes = createUploadedFile('notes.md', 'text/markdown', {
	textContent: sampleNotes
});

export const uploadedPdf = createUploadedFile('attention-is-all-you-need.pdf', 'application/pdf', {
	size: 2_215_244,
	textContent: samplePdfText
});

export const uploadedAudio = createUploadedFile('voice-memo.wav', 'audio/wav', {
	preview: toneAudioUrl,
	size: 8_078
});

export const uploadedVideo = createUploadedFile('screen-recording.webm', 'video/webm', {
	preview: clipVideoUrl,
	size: 8_101
});

export const uploadedArchive = createUploadedFile('dataset.zip', 'application/zip', {
	size: 5_242_880
});

export const uploadedMcpPrompt = createUploadedFile('summarize-issue', SpecialFileType.MCP_PROMPT, {
	mcpPrompt: {
		arguments: { issue: '#31273' },
		promptName: 'summarize-issue',
		serverName: 'github'
	},
	textContent: 'Summarize the discussion in issue #31273 and list the open questions.'
});

export const uploadedMcpPromptLoading = createUploadedFile(
	'triage-bug',
	SpecialFileType.MCP_PROMPT,
	{
		isLoading: true,
		mcpPrompt: { arguments: { label: 'bug' }, promptName: 'triage-bug', serverName: 'github' }
	}
);

export const uploadedMcpPromptError = createUploadedFile(
	'release-notes',
	SpecialFileType.MCP_PROMPT,
	{
		loadError: 'MCP server "github" is not connected',
		mcpPrompt: { promptName: 'release-notes', serverName: 'github' }
	}
);

// Conversations for the tab strip

export const tabConversations: DatabaseConversation[] = [
	{ currNode: 'm-1', id: 'conv-1', lastModified: Date.now() - 60_000, name: 'Rust borrow checker' },
	{ currNode: 'm-2', id: 'conv-2', lastModified: Date.now() - 120_000, name: 'Trip to Kyoto' },
	{ currNode: 'm-3', id: 'conv-3', lastModified: Date.now() - 180_000, name: 'GGUF quantization' },
	{
		currNode: 'm-4',
		id: 'conv-4',
		lastModified: Date.now() - 240_000,
		name: 'A very long conversation title that has to be truncated inside the tab'
	},
	{ currNode: 'm-5', id: 'conv-5', lastModified: Date.now() - 300_000, name: 'Sourdough starter' },
	{
		currNode: 'm-6',
		id: 'conv-6',
		lastModified: Date.now() - 360_000,
		name: 'SQL window functions'
	},
	{
		currNode: 'm-7',
		id: 'conv-7',
		lastModified: Date.now() - 420_000,
		name: 'Weekly report draft'
	},
	{ currNode: 'm-8', id: 'conv-8', lastModified: Date.now() - 480_000, name: 'Svelte 5 runes' },
	{ currNode: 'm-9', id: 'conv-9', lastModified: Date.now() - 540_000, name: 'Bike maintenance' },
	{ currNode: 'm-10', id: 'conv-10', lastModified: Date.now() - 600_000, name: 'Tax questions' },
	{ currNode: 'm-11', id: 'conv-11', lastModified: Date.now() - 660_000, name: 'Piano practice' },
	{ currNode: 'm-12', id: 'conv-12', lastModified: Date.now() - 720_000, name: 'CUDA kernels' }
];
