// Guards the media of a stored tool result: later requests send it as the run that
// called the tool did, so the model keeps seeing it and the server keeps its cache.

import { AttachmentType, ContentPartType, MessageRole, MessageType } from '#lib/enums/index.ts';
import { ChatService } from '#lib/services/chat.service.ts';
import type { DatabaseMessage, DatabaseMessageExtra } from '#lib/types/index.ts';
import { withoutRepeatedAttachments } from '#lib/utils/media-content-parts.ts';
import { describe, expect, it } from 'vitest';

const IMAGE_URL = 'data:image/png;base64,AAAA';
const shot: DatabaseMessageExtra = {
	base64Url: IMAGE_URL,
	name: 'mcp-attachment-1-0.png',
	type: AttachmentType.IMAGE
};

function toolResult(extra: DatabaseMessageExtra[]): DatabaseMessage {
	return {
		children: [],
		content: 'File: /tmp/shot.png',
		convId: 'c1',
		extra,
		id: 'm1',
		parent: null,
		role: MessageRole.TOOL,
		timestamp: 0,
		toolCallId: 'call_0',
		type: MessageType.TEXT
	} as DatabaseMessage;
}

describe('a stored tool result with an image', () => {
	it('sends the text and the image to a model with vision', async () => {
		const result = await ChatService.convertDbMessageToApiChatMessageData(toolResult([shot]), {
			visionInput: true
		});

		expect(result).toMatchObject({
			content: [
				{ text: 'File: /tmp/shot.png', type: ContentPartType.TEXT },
				{ type: ContentPartType.IMAGE_URL }
			],
			role: MessageRole.TOOL,
			tool_call_id: 'call_0'
		});
	});

	it('sends the text alone to a model without vision or with unknown modalities', async () => {
		for (const options of [{ visionInput: false }, {}]) {
			const result = await ChatService.convertDbMessageToApiChatMessageData(
				toolResult([shot]),
				options
			);

			expect(result.content).toBe('File: /tmp/shot.png');
		}
	});

	it('sends a repeated copy of the image once', async () => {
		const result = await ChatService.convertDbMessageToApiChatMessageData(
			toolResult([shot, { ...shot }]),
			{ visionInput: true }
		);

		expect(result.content).toHaveLength(2);
	});
});

describe('withoutRepeatedAttachments', () => {
	it('keeps attachments that share a name but not the bytes', () => {
		const other = { ...shot, base64Url: 'data:image/png;base64,BBBB' };

		expect(withoutRepeatedAttachments([shot, { ...shot }, other])).toEqual([shot, other]);
	});
});
