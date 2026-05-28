import { AttachmentType } from '$lib/enums';
import type { DatabaseMessageExtra } from '$lib/types';

function getArtifactSignature(attachment: DatabaseMessageExtra): string {
	if (attachment.artifactId) return `artifact:${attachment.artifactId}`;

	if (
		attachment.type === AttachmentType.TEXT ||
		attachment.type === AttachmentType.LEGACY_CONTEXT
	) {
		return `text:${attachment.name}:${attachment.size ?? 0}:${'content' in attachment ? attachment.content : ''}`;
	}

	if (attachment.type === AttachmentType.IMAGE && 'base64Url' in attachment) {
		return `image:${attachment.name}:${attachment.base64Url}`;
	}

	if (
		(attachment.type === AttachmentType.PDF ||
			attachment.type === AttachmentType.AUDIO ||
			attachment.type === AttachmentType.VIDEO) &&
		'base64Data' in attachment
	) {
		return `${attachment.type}:${attachment.name}:${attachment.base64Data}`;
	}

	return `${attachment.type}:${attachment.name}:${attachment.size ?? 0}`;
}

export function dedupeArtifactAttachments(
	attachments: DatabaseMessageExtra[]
): DatabaseMessageExtra[] {
	const seen = new Set<string>();
	const unique: DatabaseMessageExtra[] = [];

	for (const attachment of attachments) {
		if (attachment.presentation !== 'artifact') continue;

		const key = getArtifactSignature(attachment);
		if (seen.has(key)) continue;

		seen.add(key);
		unique.push(attachment);
	}

	return unique;
}

export function getArtifactAttachmentKey(
	messageId: string,
	attachment: DatabaseMessageExtra,
	index: number
): string {
	return `${messageId}:${getArtifactSignature(attachment)}:${index}`;
}
