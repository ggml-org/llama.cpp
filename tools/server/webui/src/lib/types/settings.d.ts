import type { SETTING_CONFIG_DEFAULT } from '$lib/constants/settings-config';
import type { ChatMessageTimings } from './chat';

export type SettingsConfigValue = string | number | boolean | null;

export interface SettingsFieldConfig {
	key: string;
	label: string;
	type: 'input' | 'textarea' | 'checkbox' | 'select';
	isExperimental?: boolean;
	help?: string;
	options?: Array<{ value: string; label: string; icon?: typeof import('@lucide/svelte').Icon }>;
}

export interface SettingsChatServiceOptions {
	stream?: boolean;
	// Model (required in ROUTER mode, optional in MODEL mode)
	model?: string;
	// System message to inject
	systemMessage?: string;
	// Disable reasoning format (use 'none' instead of 'auto')
	disableReasoningFormat?: boolean;
	// Generation parameters
	temperature?: number | null;
	max_tokens?: number | null;
	// Sampling parameters
	dynatemp_range?: number | null;
	dynatemp_exponent?: number | null;
	top_k?: number | null;
	top_p?: number | null;
	min_p?: number | null;
	xtc_probability?: number | null;
	xtc_threshold?: number | null;
	typ_p?: number | null;
	// Penalty parameters
	repeat_last_n?: number | null;
	repeat_penalty?: number | null;
	presence_penalty?: number | null;
	frequency_penalty?: number | null;
	dry_multiplier?: number | null;
	dry_base?: number | null;
	dry_allowed_length?: number | null;
	dry_penalty_last_n?: number | null;
	// Sampler configuration
	samplers?: string | string[] | null;
	// Custom parameters
	custom?: string;
	timings_per_token?: boolean;
	// Callbacks
	onChunk?: (chunk: string) => void;
	onReasoningChunk?: (chunk: string) => void;
	onToolCallChunk?: (chunk: string) => void;
	onModel?: (model: string) => void;
	onTimings?: (timings: ChatMessageTimings, promptProgress?: ChatMessagePromptProgress) => void;
	onComplete?: (
		response: string,
		reasoningContent?: string,
		timings?: ChatMessageTimings,
		toolCalls?: string
	) => void;
	onError?: (error: Error) => void;
}

export type SettingsConfigType = typeof SETTING_CONFIG_DEFAULT & {
	[key: string]: SettingsConfigValue;
};
