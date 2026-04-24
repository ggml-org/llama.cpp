import { getContext, setContext } from 'svelte';
import { CONTEXT_KEY_ATTACHMENT_PREVIEW } from '$lib/constants';

export interface AttachmentPreviewContext {
	previewItem: ChatAttachmentPreviewItem | null;
	previewOpen: boolean;
	openPreview: (item: ChatAttachmentPreviewItem, event?: Event) => void;
	closePreview: () => void;
}

const ATTACHMENT_PREVIEW_KEY = Symbol.for(CONTEXT_KEY_ATTACHMENT_PREVIEW);

export function setAttachmentPreviewContext(
	ctx: AttachmentPreviewContext
): AttachmentPreviewContext {
	return setContext(ATTACHMENT_PREVIEW_KEY, ctx);
}

export function getAttachmentPreviewContext(): AttachmentPreviewContext {
	return getContext(ATTACHMENT_PREVIEW_KEY);
}
