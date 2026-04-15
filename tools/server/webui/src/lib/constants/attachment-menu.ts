import type { Component } from 'svelte';
import { MessageSquare, Zap, FolderOpen } from '@lucide/svelte';
import { FILE_TYPE_ICONS } from '$lib/constants/icons';

export interface AttachmentMenuItem {
	/** Unique identifier for the item */
	id: string;
	/** Display label */
	label: string;
	/** Lucide icon component */
	icon: Component;
	/** Extra CSS class applied to the item (e.g. for test selectors) */
	class?: string;
	/** Whether the item requires a specific modality to be enabled */
	enabledWhen?: 'always' | 'hasVisionModality' | 'hasAudioModality';
	/** Tooltip shown when the item is disabled */
	disabledTooltip?: string;
	/** Callback key on the Props interface to invoke when clicked */
	action: 'onFileUpload' | 'onSystemPromptClick' | 'onMcpPromptClick' | 'onMcpResourcesClick';
	/** Whether the item is only shown when a specific capability is present */
	visibleWhen?: 'hasMcpPromptsSupport' | 'hasMcpResourcesSupport';
	/** Whether this item has a tooltip even when enabled (uses dynamic text) */
	hasEnabledTooltip?: boolean;
}

/**
 * File attachment menu items shown in both the desktop dropdown and mobile sheet.
 * The "Tools" submenu is handled separately by each component.
 */
export const ATTACHMENT_FILE_ITEMS: AttachmentMenuItem[] = [
	{
		id: 'images',
		label: 'Images',
		icon: FILE_TYPE_ICONS.image,
		class: 'images-button',
		enabledWhen: 'hasVisionModality',
		disabledTooltip: 'Image processing requires a vision model',
		action: 'onFileUpload'
	},
	{
		id: 'audio',
		label: 'Audio Files',
		icon: FILE_TYPE_ICONS.audio,
		class: 'audio-button',
		enabledWhen: 'hasAudioModality',
		disabledTooltip: 'Audio files processing requires an audio model',
		action: 'onFileUpload'
	},
	{
		id: 'text',
		label: 'Text Files',
		icon: FILE_TYPE_ICONS.text,
		enabledWhen: 'always',
		action: 'onFileUpload'
	},
	{
		id: 'pdf',
		label: 'PDF Files',
		icon: FILE_TYPE_ICONS.pdf,
		enabledWhen: 'always',
		disabledTooltip: 'PDFs will be converted to text. Image-based PDFs may not work properly.',
		hasEnabledTooltip: true,
		action: 'onFileUpload'
	}
];

export const ATTACHMENT_EXTRA_ITEMS: AttachmentMenuItem[] = [
	{
		id: 'system-message',
		label: 'System Message',
		icon: MessageSquare,
		enabledWhen: 'always',
		hasEnabledTooltip: true,
		action: 'onSystemPromptClick'
	}
];

export const ATTACHMENT_MCP_ITEMS: AttachmentMenuItem[] = [
	{
		id: 'mcp-prompt',
		label: 'MCP Prompt',
		icon: Zap,
		enabledWhen: 'always',
		action: 'onMcpPromptClick',
		visibleWhen: 'hasMcpPromptsSupport'
	},
	{
		id: 'mcp-resources',
		label: 'MCP Resources',
		icon: FolderOpen,
		enabledWhen: 'always',
		action: 'onMcpResourcesClick',
		visibleWhen: 'hasMcpResourcesSupport'
	}
];

export const ATTACHMENT_TOOLTIP_TEXT = 'Add files, system prompt or MCP Servers';
