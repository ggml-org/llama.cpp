/**
 * Icon mappings for file types and model modalities
 * Centralized configuration to ensure consistent icon usage across the app
 */

import {
	File as FileIcon,
	FileText as FileTextIcon,
	Image as ImageIcon,
	Eye as VisionIcon,
	Mic as AudioIcon,
	Video as VideoIcon
} from '@lucide/svelte';
import { FileTypeCategory, ModelModality } from '$lib/enums';

export const FILE_TYPE_ICONS = {
	[FileTypeCategory.IMAGE]: ImageIcon,
	[FileTypeCategory.AUDIO]: AudioIcon,
	[FileTypeCategory.VIDEO]: VideoIcon,
	[FileTypeCategory.TEXT]: FileTextIcon,
	[FileTypeCategory.PDF]: FileIcon
} as const;

export const DEFAULT_FILE_ICON = FileIcon;

export const MODALITY_ICONS = {
	[ModelModality.VISION]: VisionIcon,
	[ModelModality.AUDIO]: AudioIcon,
	[ModelModality.VIDEO]: VideoIcon
} as const;

export const MODALITY_LABELS = {
	[ModelModality.VISION]: 'Vision',
	[ModelModality.AUDIO]: 'Audio',
	[ModelModality.VIDEO]: 'Video'
} as const;

// Shared SVG icon strings for copy and preview buttons
export const COPY_ICON_SVG = `<svg xmlns="http://www.w3.org/2000/svg" width="16" height="16" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2" stroke-linecap="round" stroke-linejoin="round" class="lucide lucide-copy-icon lucide-copy"><rect width="14" height="14" x="8" y="8" rx="2" ry="2"/><path d="M4 16c-1.1 0-2-.9-2-2V4c0-1.1.9-2 2-2h10c1.1 0 2 .9 2 2"/></svg>`;

export const PREVIEW_ICON_SVG = `<svg xmlns="http://www.w3.org/2000/svg" width="16" height="16" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2" stroke-linecap="round" stroke-linejoin="round" class="lucide lucide-eye lucide-eye-icon"><path d="M2.062 12.345a1 1 0 0 1 0-.69C3.5 7.73 7.36 5 12 5s8.5 2.73 9.938 6.655a1 1 0 0 1 0 .69C20.5 16.27 16.64 19 12 19s-8.5-2.73-9.938-6.655"/><circle cx="12" cy="12" r="3"/></svg>`;
