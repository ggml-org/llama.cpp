import type { SETTING_CONFIG_DEFAULT } from '$lib/constants/settings-config';
import type { ChatMessageTimings } from './chat';

export type SettingsConfigValue = string | number | boolean;

export interface SettingsFieldConfig {
	key: string;
	label: string;
	type: 'input' | 'textarea' | 'checkbox' | 'select';
	help?: string;
	options?: Array<{ value: string; label: string; icon?: typeof import('@lucide/svelte').Icon }>;
}

export interface SettingsChatServiceOptions {
	stream?: boolean;
	// Generation parameters
	temperature?: number;
	max_tokens?: number;
	// Sampling parameters
	dynatemp_range?: number;
	dynatemp_exponent?: number;
	top_k?: number;
	top_p?: number;
	min_p?: number;
	xtc_probability?: number;
	xtc_threshold?: number;
	typ_p?: number;
	// Penalty parameters
	repeat_last_n?: number;
	repeat_penalty?: number;
	presence_penalty?: number;
	frequency_penalty?: number;
	dry_multiplier?: number;
	dry_base?: number;
	dry_allowed_length?: number;
	dry_penalty_last_n?: number;
	// Sampler configuration
	samplers?: string | string[];
	// Custom parameters
	custom?: string;
	// Callbacks
	onChunk?: (chunk: string) => void;
	onReasoningChunk?: (chunk: string) => void;
	onComplete?: (response: string, reasoningContent?: string, timings?: ChatMessageTimings) => void;
	onError?: (error: Error) => void;
}

export type SettingsConfigType = typeof SETTING_CONFIG_DEFAULT & {
	[key: string]: SettingsConfigValue;
};
