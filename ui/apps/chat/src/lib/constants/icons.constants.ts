/**
 * Icon and modality mappings for file types and models
 * Centralized configuration to ensure consistent icon usage across the app
 */

import { MODALITY_FLAG_KEYS, MODALITY_ORDER } from '@ggml-org/llama-ui/models';
import {
	CircleAlert as LoadFailedIcon,
	Compass as DiscoverIcon,
	Eject as UnloadIcon,
	File as FileIcon,
	FileText as FileTextIcon,
	FolderOpen as CwdCommandIcon,
	FoldVertical as CompactCommandIcon,
	Image as ImageIcon,
	Loader2 as LoadingIcon,
	Mic as AudioIcon,
	Package as ModelIcon,
	Pause as PauseDownloadIcon,
	Play as ResumeDownloadIcon,
	Power as LoadIcon,
	RotateCw as RetryIcon,
	Sparkles as PromptCommandIcon,
	Video as VideoIcon
} from '@lucide/svelte';
import { ChatFormCommandAction, FileTypeCategory, ModelCapability } from '#lib/enums/index.ts';
import type { ModelCapabilities } from '#lib/types/models.js';
import type { Component } from 'svelte';
export {
	CAPABILITY_ICONS,
	CAPABILITY_LABELS,
	MODALITY_FLAG_KEYS,
	MODALITY_ICONS,
	MODALITY_LABELS,
	MODALITY_ORDER
} from '@ggml-org/llama-ui/models';

export const FILE_TYPE_ICONS = {
	[FileTypeCategory.AUDIO]: AudioIcon,
	[FileTypeCategory.IMAGE]: ImageIcon,
	[FileTypeCategory.PDF]: FileIcon,
	[FileTypeCategory.TEXT]: FileTextIcon,
	[FileTypeCategory.VIDEO]: VideoIcon
} as const;

export const DEFAULT_FILE_ICON = FileIcon;

/** The mark for a model, wherever one is listed. */
export const MODEL_ICON = ModelIcon;

/** The mark for the Discover Models surface and its entry points. */
export const DISCOVER_ICON = DiscoverIcon;

/** Boolean flag a modality drives on the ModelModalities type. */
export type ModalityKey = (typeof MODALITY_FLAG_KEYS)[keyof typeof MODALITY_FLAG_KEYS];

/** Modality flags a model can carry, in display order. */
export const MODALITY_KEYS: ModalityKey[] = MODALITY_ORDER.map(
	(modality) => MODALITY_FLAG_KEYS[modality]
);

/** Maps a ModelCapability to the boolean flag it drives on the ModelCapabilities type */
export const CAPABILITY_FLAG_KEYS: Record<ModelCapability, keyof ModelCapabilities> = {
	[ModelCapability.REASONING]: 'reasoning',
	[ModelCapability.TOOL_USE]: 'tools'
};

/** Icons of the slash-command picker, one per chat-form command. */
export const CHAT_FORM_COMMAND_ICONS: Record<ChatFormCommandAction, Component> = {
	[ChatFormCommandAction.COMPACT]: CompactCommandIcon,
	[ChatFormCommandAction.CWD]: CwdCommandIcon,
	[ChatFormCommandAction.MODEL]: MODEL_ICON,
	[ChatFormCommandAction.PROMPT]: PromptCommandIcon
};

/** Icons of the download control a manager row shows while it downloads. */
export const MODEL_DOWNLOAD_ICONS = {
	pause: PauseDownloadIcon,
	resume: ResumeDownloadIcon
} as const;

/** Icons the model load control shows for its states and actions. */
export const MODEL_LOAD_ICONS = {
	failed: LoadFailedIcon,
	load: LoadIcon,
	loading: LoadingIcon,
	retry: RetryIcon,
	unload: UnloadIcon
} as const;

// Shared SVG icon strings for copy and preview buttons
export const COPY_ICON_SVG = `<svg xmlns="http://www.w3.org/2000/svg" width="16" height="16" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2" stroke-linecap="round" stroke-linejoin="round" class="lucide lucide-copy-icon lucide-copy"><rect width="14" height="14" x="8" y="8" rx="2" ry="2"/><path d="M4 16c-1.1 0-2-.9-2-2V4c0-1.1.9-2 2-2h10c1.1 0 2 .9 2 2"/></svg>`;

export const PREVIEW_ICON_SVG = `<svg xmlns="http://www.w3.org/2000/svg" width="16" height="16" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2" stroke-linecap="round" stroke-linejoin="round" class="lucide lucide-eye lucide-eye-icon"><path d="M2.062 12.345a1 1 0 0 1 0-.69C3.5 7.73 7.36 5 12 5s8.5 2.73 9.938 6.655a1 1 0 0 1 0 .69C20.5 16.27 16.64 19 12 19s-8.5-2.73-9.938-6.655"/><circle cx="12" cy="12" r="3"/></svg>`;

export const CODE_ICON_SVG = `<svg xmlns="http://www.w3.org/2000/svg" width="16" height="16" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2" stroke-linecap="round" stroke-linejoin="round" class="lucide lucide-code lucide-code-icon"><path d="m16 18 6-6-6-6"/><path d="m8 6-6 6 6 6"/></svg>`;
