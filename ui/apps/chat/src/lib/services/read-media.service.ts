/**
 * ReadMediaService - Reads local media files for the read_media tool
 *
 * Encodes image and audio files as base64 data URLs with the metadata the
 * model needs. No reactive state; consumed by toolsStore.
 */

import { ToolsService } from './tools.service';
import {
	BASE64_IMAGE_URI_REGEX,
	FILE_EXTENSION_SEPARATOR,
	FILE_PATH_SEPARATOR_REGEX,
	NEWLINE,
	PREFIX_FILE,
	PREFIX_MIME,
	PREFIX_SIZE,
	READ_MEDIA_ATTACHMENT_PREFIX,
	READ_MEDIA_AUDIO_MIME,
	READ_MEDIA_IMAGE_MIME,
	RESP_TYPE_BASE64
} from '#lib/constants/index.ts';
import { BuiltInTool, ToolResponseField } from '#lib/enums/index.ts';
import type { DatabaseMessage, ToolExecutionResult } from '#lib/types/index.ts';
import { findReferencedImage } from '#lib/utils/image-eviction.ts';

/** Modalities of the model the tool call runs for. */
export interface ReadMediaCapabilities {
	audio: boolean;
	vision: boolean;
}

/** Lowercase extension of a path, without the dot. Empty when the file name has none. */
function fileExtension(path: string): string {
	const name = path.split(FILE_PATH_SEPARATOR_REGEX).pop() ?? '';
	const dot = name.lastIndexOf(FILE_EXTENSION_SEPARATOR);

	return dot > 0 ? name.slice(dot + 1).toLowerCase() : '';
}

/**
 * **ReadMediaService** - browser executor for the `read_media` tool
 *
 * The tool is synthetic: no such tool exists on the server. It reads the file
 * through the server `read_file` tool with the `base64` response type, then
 * turns the bytes into a data URI line. The agentic store lifts that line into
 * an image or audio attachment on the tool result message, which is what makes
 * the model perceive the file instead of reading a wall of base64.
 *
 * An `attachment:` path names an image attached to the conversation. The
 * browser holds its bytes, so no call goes to the server.
 *
 * Living in the browser is what lets it exist only for models that can
 * actually use the result - the server has no idea which model is selected.
 *
 * @see buildReadMediaToolDefinition in constants/read-media.ts - tool schema sent to the LLM
 * @see agenticStore in stores/agentic/index.svelte.ts - tool dispatch and attachment extraction
 */
export class ReadMediaService {
	static async executeTool(
		params: Record<string, unknown>,
		capabilities: ReadMediaCapabilities,
		signal?: AbortSignal,
		cwd?: string,
		loadMessages?: () => Promise<DatabaseMessage[]>
	): Promise<ToolExecutionResult> {
		const path = typeof params.path === 'string' ? params.path : '';

		if (!path) {
			return { content: 'Error: missing "path" argument.', isError: true };
		}

		if (path.startsWith(READ_MEDIA_ATTACHMENT_PREFIX)) {
			return ReadMediaService.readAttachment(path, capabilities, (await loadMessages?.()) ?? []);
		}

		const extension = fileExtension(path);
		const imageMime = READ_MEDIA_IMAGE_MIME[extension];
		const audioMime = READ_MEDIA_AUDIO_MIME[extension];

		let resolvedMime: string | undefined;

		if (imageMime && capabilities.vision) resolvedMime = imageMime;
		else if (audioMime && capabilities.audio) resolvedMime = audioMime;

		if (!resolvedMime) {
			const supported = [
				...(capabilities.vision ? Object.keys(READ_MEDIA_IMAGE_MIME) : []),
				...(capabilities.audio ? Object.keys(READ_MEDIA_AUDIO_MIME) : [])
			];
			// an unreadable-by-this-model file is a dead end, so say why instead of failing silently
			const reason =
				imageMime || audioMime
					? `the current model cannot perceive ".${extension}" files`
					: `".${extension}" is not a supported media type`;

			return {
				content: `Error: ${reason}. Supported: ${supported.join(', ')}.`,
				isError: true
			};
		}

		const raw = await ToolsService.executeToolRaw(
			BuiltInTool.SERVER_READ_FILE,
			{ path },
			signal,
			cwd,
			RESP_TYPE_BASE64
		);

		if (ToolResponseField.ERROR in raw) {
			return { content: String(raw[ToolResponseField.ERROR]), isError: true };
		}

		const base64 = typeof raw.base64 === 'string' ? raw.base64 : '';

		if (!base64) {
			return { content: `Error: no data returned for ${path}.`, isError: true };
		}

		const sizeBytes = typeof raw.size_bytes === 'number' ? raw.size_bytes : 0;
		const content = [
			`${PREFIX_FILE}${path}`,
			`${PREFIX_SIZE}${sizeBytes} bytes`,
			`${PREFIX_MIME}${resolvedMime}`,
			`data:${resolvedMime};base64,${base64}`
		].join(NEWLINE);

		return { content, isError: false };
	}

	/** An image attached to the conversation, in the same result shape as a file. */
	private static readAttachment(
		reference: string,
		capabilities: ReadMediaCapabilities,
		messages: DatabaseMessage[]
	): ToolExecutionResult {
		if (!capabilities.vision) {
			return { content: 'Error: the current model cannot perceive images.', isError: true };
		}

		const image = findReferencedImage(reference, messages);
		const mimeType = image?.base64Url.match(BASE64_IMAGE_URI_REGEX)?.[1];

		if (!image || !mimeType) {
			return { content: `Error: no image attachment found for ${reference}.`, isError: true };
		}

		const content = [
			`${PREFIX_FILE}${image.name}`,
			`${PREFIX_SIZE}${image.size ?? 0} bytes`,
			`${PREFIX_MIME}${mimeType}`,
			image.base64Url
		].join(NEWLINE);

		return { content, isError: false };
	}
}
