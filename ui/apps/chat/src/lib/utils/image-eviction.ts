import {
	buildEvictedImageNote,
	NEWLINE,
	READ_MEDIA_ATTACHMENT_ID_LENGTH,
	READ_MEDIA_ATTACHMENT_PREFIX,
	READ_MEDIA_ATTACHMENT_REGEX
} from '#lib/constants/index.ts';
import { AttachmentType, MessageRole, MessageType } from '#lib/enums/index.ts';
import type { ApiChatMessageData } from '#lib/types/api.js';
import type { DatabaseMessage, DatabaseMessageExtraImageFile } from '#lib/types/database.js';
import { withoutRepeatedAttachments } from '#lib/utils/media-content-parts.ts';

type ContextMessage = ApiChatMessageData | DatabaseMessage;

// the roles whose messages hold images the model reads
const MEDIA_ROLES: ReadonlySet<string> = new Set([MessageRole.USER, MessageRole.TOOL]);

function isStored(message: ContextMessage): message is DatabaseMessage {
	return 'convId' in message;
}

function attachmentReference(message: DatabaseMessage, index: number): string {
	const idStart = message.id.slice(0, READ_MEDIA_ATTACHMENT_ID_LENGTH);

	return `${READ_MEDIA_ATTACHMENT_PREFIX}${idStart}/${index + 1}`;
}

/** The image an attachment reference names, among the messages of its conversation. */
export function findReferencedImage(
	reference: string,
	messages: DatabaseMessage[]
): DatabaseMessageExtraImageFile | undefined {
	const match = reference.trim().match(READ_MEDIA_ATTACHMENT_REGEX);

	if (!match) return undefined;

	const [, idStart, position] = match;
	const message = messages.find((m) => m.id.slice(0, READ_MEDIA_ATTACHMENT_ID_LENGTH) === idStart);
	const extra = message?.extra?.[Number(position) - 1];

	return extra?.type === AttachmentType.IMAGE ? extra : undefined;
}

function withImageNotes(message: DatabaseMessage): DatabaseMessage {
	const notes: string[] = [];
	const extra = [];
	const kept = new Set(withoutRepeatedAttachments(message.extra ?? []));

	for (const [index, attachment] of (message.extra ?? []).entries()) {
		if (attachment.type === AttachmentType.IMAGE) {
			// a repeated copy is the same image, one note covers it
			if (kept.has(attachment)) {
				notes.push(buildEvictedImageNote(attachment.name, attachmentReference(message, index)));
			}
		} else {
			extra.push(attachment);
		}
	}

	if (notes.length === 0) return message;

	return {
		...message,
		content: [message.content, ...notes].filter(Boolean).join(NEWLINE),
		extra
	};
}

/**
 * Takes the images out of the turns that are far behind: the user's attachments and
 * the images of tool results, read_media's own included. A note stays in
 * the place of each one and tells the model how to load it again with read_media.
 * Only the copy sent to the model changes, the stored messages keep their images.
 *
 * The cut moves in steps of `keepTurns`, so between `keepTurns` and
 * `2 * keepTurns - 1` turns keep their images. A cut that moved every turn
 * would change the prompt every turn and the server could not reuse its cache.
 */
export function evictOldImages<T extends ContextMessage>(messages: T[], keepTurns: number): T[] {
	const turnStarts: number[] = [];

	messages.forEach((m, i) => {
		const isCompaction = isStored(m) && m.type === MessageType.COMPACTION;

		if (m.role === MessageRole.USER && !isCompaction) turnStarts.push(i);
	});

	const steps = Math.floor(turnStarts.length / keepTurns) - 1;

	if (steps < 1) return messages;

	const cut = turnStarts[steps * keepTurns];

	return messages.map((m, i) =>
		i < cut && isStored(m) && MEDIA_ROLES.has(m.role) ? (withImageNotes(m) as T) : m
	);
}
