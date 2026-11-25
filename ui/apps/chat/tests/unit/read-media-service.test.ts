import { AttachmentType, MessageRole, MessageType } from '#lib/enums/index.ts';
import { ReadMediaService } from '#lib/services/read-media.service.ts';
import { ToolsService } from '#lib/services/tools.service.ts';
import type { DatabaseMessage } from '#lib/types/index.ts';
import { beforeEach, describe, expect, it, vi } from 'vitest';

const IMAGE_URL = 'data:image/png;base64,AAAA';
const VISION = { audio: false, vision: true };
const MESSAGES = [
	{
		children: [],
		content: 'look',
		convId: 'conv',
		extra: [{ base64Url: IMAGE_URL, name: 'cat.png', size: 3, type: AttachmentType.IMAGE }],
		id: 'user-0-id',
		parent: null,
		role: MessageRole.USER,
		timestamp: 0,
		type: MessageType.TEXT
	}
] as DatabaseMessage[];

function readAttachment(path: string, capabilities = VISION) {
	return ReadMediaService.executeTool({ path }, capabilities, undefined, undefined, async () => [
		...MESSAGES
	]);
}

beforeEach(() => {
	vi.restoreAllMocks();
});

describe('ReadMediaService attachment references', () => {
	it('returns an attached image without a call to the server', async () => {
		const executeToolRaw = vi.spyOn(ToolsService, 'executeToolRaw');
		const result = await readAttachment('attachment:user-0-i/1');

		expect(result).toEqual({
			content: `File: cat.png\nSize: 3 bytes\nMIME: image/png\n${IMAGE_URL}`,
			isError: false
		});
		expect(executeToolRaw).not.toHaveBeenCalled();
	});

	it('reports a reference that names no image', async () => {
		const result = await readAttachment('attachment:user-0-i/2');

		expect(result).toEqual({
			content: 'Error: no image attachment found for attachment:user-0-i/2.',
			isError: true
		});
	});

	it('refuses an image for a model without vision', async () => {
		const result = await readAttachment('attachment:user-0-i/1', { audio: true, vision: false });

		expect(result.isError).toBe(true);
	});
});
