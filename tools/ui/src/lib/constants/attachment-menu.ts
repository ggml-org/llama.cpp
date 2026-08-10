import { FolderOpen, MessageSquare, Zap } from '@lucide/svelte';
import { FILE_TYPE_ICONS } from '$lib/constants/icons';
import {
	AttachmentAction,
	AttachmentItemEnabledWhen,
	AttachmentItemVisibleWhen,
	AttachmentMenuItemId
} from '$lib/enums';
import type { Component } from 'svelte';

export interface AttachmentMenuItem {
	/** Unique identifier for the item */
	id: AttachmentMenuItemId;
	/** Display label */
	label: string;
	/** Lucide icon component */
	icon: Component;
	/** Extra CSS class applied to the item (e.g. for test selectors) */
	class?: string;
	/** Whether the item requires a specific modality to be enabled */
	enabledWhen?: AttachmentItemEnabledWhen;
	/** Tooltip shown when the item is disabled */
	disabledTooltip?: string;
	/** Callback key on the Props interface to invoke when clicked */
	action: AttachmentAction;
	/** Whether the item is only shown when a specific capability is present */
	visibleWhen?: AttachmentItemVisibleWhen;
	/** Whether this item has a tooltip even when enabled (uses dynamic text) */
	hasEnabledTooltip?: boolean;
}

/**
 * File attachment menu items shown in both the desktop dropdown and mobile sheet.
 * The "Tools" submenu is handled separately by each component.
 */
export const ATTACHMENT_FILE_ITEMS: AttachmentMenuItem[] = [
	{
		action: AttachmentAction.FILE_UPLOAD,
		class: 'images-button',
		disabledTooltip: 'Image processing requires a vision model',
		enabledWhen: AttachmentItemEnabledWhen.HAS_VISION_MODALITY,
		icon: FILE_TYPE_ICONS.image,
		id: AttachmentMenuItemId.IMAGES,
		label: 'Images'
	},
	{
		action: AttachmentAction.FILE_UPLOAD,
		class: 'audio-button',
		disabledTooltip: 'Audio files processing requires an audio model',
		enabledWhen: AttachmentItemEnabledWhen.HAS_AUDIO_MODALITY,
		icon: FILE_TYPE_ICONS.audio,
		id: AttachmentMenuItemId.AUDIO,
		label: 'Audio Files'
	},
	{
		action: AttachmentAction.FILE_UPLOAD,
		class: 'video-button',
		disabledTooltip: 'Video files processing requires a video model',
		enabledWhen: AttachmentItemEnabledWhen.HAS_VIDEO_MODALITY,
		icon: FILE_TYPE_ICONS.video,
		id: AttachmentMenuItemId.VIDEO,
		label: 'Video Files'
	},
	{
		action: AttachmentAction.FILE_UPLOAD,
		enabledWhen: AttachmentItemEnabledWhen.ALWAYS,
		icon: FILE_TYPE_ICONS.text,
		id: AttachmentMenuItemId.TEXT,
		label: 'Text Files'
	},
	{
		action: AttachmentAction.FILE_UPLOAD,
		disabledTooltip: 'PDFs will be converted to text. Image-based PDFs may not work properly.',
		enabledWhen: AttachmentItemEnabledWhen.ALWAYS,
		hasEnabledTooltip: true,
		icon: FILE_TYPE_ICONS.pdf,
		id: AttachmentMenuItemId.PDF,
		label: 'PDF Files'
	}
];

export const ATTACHMENT_EXTRA_ITEMS: AttachmentMenuItem[] = [];

export const ATTACHMENT_PROMPT_ITEMS: AttachmentMenuItem[] = [
	{
		action: AttachmentAction.SYSTEM_PROMPT_CLICK,
		enabledWhen: AttachmentItemEnabledWhen.ALWAYS,
		hasEnabledTooltip: true,
		icon: MessageSquare,
		id: AttachmentMenuItemId.SYSTEM_MESSAGE,
		label: 'System Message'
	},
	{
		action: AttachmentAction.MCP_PROMPT_CLICK,
		enabledWhen: AttachmentItemEnabledWhen.ALWAYS,
		icon: Zap,
		id: AttachmentMenuItemId.MCP_PROMPT,
		label: 'MCP Prompt',
		visibleWhen: AttachmentItemVisibleWhen.HAS_MCP_PROMPTS_SUPPORT
	}
];

export const ATTACHMENT_MCP_ITEMS: AttachmentMenuItem[] = [
	{
		action: AttachmentAction.MCP_RESOURCES_CLICK,
		enabledWhen: AttachmentItemEnabledWhen.ALWAYS,
		icon: FolderOpen,
		id: AttachmentMenuItemId.MCP_RESOURCES,
		label: 'MCP Resources',
		visibleWhen: AttachmentItemVisibleWhen.HAS_MCP_RESOURCES_SUPPORT
	}
];

export const ATTACHMENT_TOOLTIP_TEXT = 'Add files, prompts, tools or MCP Servers';
