import type { Component } from 'svelte';
import { MessageSquare, Zap, FolderOpen } from '@lucide/svelte';
import { FILE_TYPE_ICONS } from '$lib/constants/icons';
import {
	AttachmentAction,
	AttachmentItemEnabledWhen,
	AttachmentItemVisibleWhen,
	AttachmentMenuItemId
} from '$lib/enums';

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
		id: AttachmentMenuItemId.IMAGES,
		label: 'Images',
		icon: FILE_TYPE_ICONS.image,
		class: 'images-button',
		enabledWhen: AttachmentItemEnabledWhen.HAS_VISION_MODALITY,
		disabledTooltip: 'Image processing requires a vision model',
		action: AttachmentAction.FILE_UPLOAD
	},
	{
		id: AttachmentMenuItemId.AUDIO,
		label: 'Audio Files',
		icon: FILE_TYPE_ICONS.audio,
		class: 'audio-button',
		enabledWhen: AttachmentItemEnabledWhen.HAS_AUDIO_MODALITY,
		disabledTooltip: 'Audio files processing requires an audio model',
		action: AttachmentAction.FILE_UPLOAD
	},
	{
		id: AttachmentMenuItemId.TEXT,
		label: 'Text Files',
		icon: FILE_TYPE_ICONS.text,
		enabledWhen: AttachmentItemEnabledWhen.ALWAYS,
		action: AttachmentAction.FILE_UPLOAD
	},
	{
		id: AttachmentMenuItemId.PDF,
		label: 'PDF Files',
		icon: FILE_TYPE_ICONS.pdf,
		enabledWhen: AttachmentItemEnabledWhen.ALWAYS,
		disabledTooltip: 'PDFs will be converted to text. Image-based PDFs may not work properly.',
		hasEnabledTooltip: true,
		action: AttachmentAction.FILE_UPLOAD
	}
];

export const ATTACHMENT_EXTRA_ITEMS: AttachmentMenuItem[] = [
	{
		id: AttachmentMenuItemId.SYSTEM_MESSAGE,
		label: 'System Message',
		icon: MessageSquare,
		enabledWhen: AttachmentItemEnabledWhen.ALWAYS,
		hasEnabledTooltip: true,
		action: AttachmentAction.SYSTEM_PROMPT_CLICK
	}
];

export const ATTACHMENT_MCP_ITEMS: AttachmentMenuItem[] = [
	{
		id: AttachmentMenuItemId.MCP_PROMPT,
		label: 'MCP Prompt',
		icon: Zap,
		enabledWhen: AttachmentItemEnabledWhen.ALWAYS,
		action: AttachmentAction.MCP_PROMPT_CLICK,
		visibleWhen: AttachmentItemVisibleWhen.HAS_MCP_PROMPTS_SUPPORT
	},
	{
		id: AttachmentMenuItemId.MCP_RESOURCES,
		label: 'MCP Resources',
		icon: FolderOpen,
		enabledWhen: AttachmentItemEnabledWhen.ALWAYS,
		action: AttachmentAction.MCP_RESOURCES_CLICK,
		visibleWhen: AttachmentItemVisibleWhen.HAS_MCP_RESOURCES_SUPPORT
	}
];

export const ATTACHMENT_TOOLTIP_TEXT = 'Add files, prompts, tools or MCP Servers';
