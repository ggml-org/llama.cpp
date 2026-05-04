/**
 * Icon mappings for file types and model modalities
 * Centralized configuration to ensure consistent icon usage across the app
 */

import {
	File as FileIcon,
	FileText as FileTextIcon,
	Image as ImageIcon,
	Eye as VisionIcon,
	Mic as AudioIcon
} from '@lucide/svelte';
import { FileTypeCategory, ModelModality } from '$lib/enums';

export const FILE_TYPE_ICONS = {
	[FileTypeCategory.IMAGE]: ImageIcon,
	[FileTypeCategory.AUDIO]: AudioIcon,
	[FileTypeCategory.TEXT]: FileTextIcon,
	[FileTypeCategory.PDF]: FileIcon
} as const;

export const DEFAULT_FILE_ICON = FileIcon;

export const MODALITY_ICONS = {
	[ModelModality.VISION]: VisionIcon,
	[ModelModality.AUDIO]: AudioIcon
} as const;

export const MODALITY_LABELS = {
	[ModelModality.VISION]: 'Vision',
	[ModelModality.AUDIO]: 'Audio'
} as const;
