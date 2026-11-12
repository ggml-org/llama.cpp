// Model UI's constants, from ui/apps/chat's src/lib/constants (css-classes, icons, ui, huggingface,
// models-discover-download and model-compatibility).
import { ModelCapability, ModelModality } from './enums.js';
import type { ModelModalities } from './types.js';
import {
	Eye as VisionIcon,
	Lightbulb as ReasoningIcon,
	Mic as AudioIcon,
	Video as VideoIcon,
	Wrench as ToolUseIcon
} from '@lucide/svelte';
import type { Component } from 'svelte';

/** Neutral model badge: params, quantization, tags. */
export const MODEL_BADGE_CLASS =
	'inline-flex w-fit shrink-0 items-center justify-center whitespace-nowrap rounded-md border border-border/50 px-1 py-0 text-[10px] font-mono bg-foreground/15 dark:bg-foreground/10 text-foreground [a&]:hover:bg-foreground/25';

/** Emphasis model badge: draft sidecars and other markers that must stand out. */
export const MODEL_VARIANT_BADGE_CLASS =
	'inline-flex w-fit shrink-0 items-center justify-center whitespace-nowrap rounded-md bg-primary px-1.5 py-0 text-[10px] font-mono font-semibold uppercase tracking-wide text-primary-foreground';

/** Tag badge: free-form tags on a model id. */
export const MODEL_TAG_BADGE_CLASS =
	'inline-flex w-fit shrink-0 items-center justify-center whitespace-nowrap rounded-md border border-border/50 px-1 py-0 text-[10px] font-mono text-foreground [a&]:hover:bg-accent [a&]:hover:text-accent-foreground';

/** A draft that rides with a quant chip, e.g. `MTP:Q8_0`. */
export const SIDECAR_BADGE_CLASS =
	'rounded-md bg-primary px-1 py-0.5 text-[10px] font-semibold tracking-wide text-primary-foreground uppercase';

export const MODALITY_ICONS = {
	[ModelModality.AUDIO]: AudioIcon,
	[ModelModality.VIDEO]: VideoIcon,
	[ModelModality.VISION]: VisionIcon
};

export const MODALITY_LABELS = {
	[ModelModality.AUDIO]: 'Audio',
	[ModelModality.VIDEO]: 'Video',
	[ModelModality.VISION]: 'Vision'
};

export const MODALITY_FLAG_KEYS: Record<Exclude<ModelModality, ModelModality.TEXT>, keyof ModelModalities> = {
	[ModelModality.AUDIO]: 'audio',
	[ModelModality.VIDEO]: 'video',
	[ModelModality.VISION]: 'vision'
};

export const MODALITY_ORDER: Exclude<ModelModality, ModelModality.TEXT>[] = [
	ModelModality.VISION,
	ModelModality.AUDIO,
	ModelModality.VIDEO
];

export const CAPABILITY_ICONS: Record<ModelCapability, Component> = {
	[ModelCapability.REASONING]: ReasoningIcon,
	[ModelCapability.TOOL_USE]: ToolUseIcon
};

export const CAPABILITY_LABELS: Record<ModelCapability, string> = {
	[ModelCapability.REASONING]: 'Reasoning',
	[ModelCapability.TOOL_USE]: 'Tool use'
};

/** Orgs whose avatar is a dark mark on transparent, inverted in dark mode. */
export const DARK_INVERT_AVATAR_ORGS = ['openai'];

export const HF_AVATARS_URL = 'https://huggingface.co/api/avatars';

/** Bit depth that collects every quant no other row takes. */
export const OTHER_BIT_DEPTH = 99;
export const OTHER_BIT_DEPTH_LABEL = 'Other';
export const BIT_DEPTH_LABEL_SUFFIX = '-bit';

export const KILOBYTE = 1_000;
export const MEGABYTE = 1_000_000;
export const GIGABYTE = 1_000_000_000;
export const BYTE_LABEL = 'B';
export const KILOBYTE_LABEL = 'KB';
export const MEGABYTE_LABEL = 'MB';
export const GIGABYTE_LABEL = 'GB';

/** Bytes in one mebibyte (MiB), used to convert a file size to MB. */
export const MIB_BYTES = 1_048_576;
/** MiB in one tier unit; the tiers below are binary sizes, i.e. GiB. */
export const MB_PER_GB = 1024;
/** Overhead multiplier applied to the file size when estimating weight memory. */
export const QUANT_WEIGHT = 1.05;
/** Share of RAM the app allows the model to occupy. */
export const RAM_BUDGET_RATIO = 0.75;
/** Fixed RAM overhead (MB) reserved for the system and KV cache. */
export const RAM_OVERHEAD_MB = 2048;
/** Memory sizes devices come with, in GB; a model is shown against the smallest it fits. */
export const MEM_TIERS = [4, 6, 8, 12, 16, 24, 32, 48, 64, 96, 128, 192, 256, 384, 512, 768, 1024];
