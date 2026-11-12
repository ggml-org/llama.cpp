// What Llama Chat's Discover knows about a few of llama.app's repos, read from the Hugging Face API on 2026-11-07:
// downloads, likes, the base model, the GGUF's context length and architecture, and the capabilities Discover works
// out itself (vision from the pipeline tag or an mmproj file, reasoning and tool use from the chat template, run
// through ui/apps/chat's own detectors). For the Discover side of the comparison.

export interface HubModel {
	id: string;
	downloads: number;
	likes: number;
	lastModified: string;
	baseModel: string | null;
	license: string | null;
	contextLength: number | null;
	params: number | null;
	architecture: string | null;
	vision: boolean;
	audio: boolean;
	video: boolean;
	reasoning: boolean;
	tools: boolean;
}

export const HUB_MODELS: HubModel[] = [
	{
		id: 'ggml-org/gpt-oss-20b-GGUF',
		downloads: 329258,
		likes: 195,
		lastModified: '2026-10-19T14:14:26.000Z',
		baseModel: 'openai/gpt-oss-20b',
		license: 'apache-2.0',
		contextLength: 131072,
		params: 20914757184,
		architecture: 'gpt-oss',
		vision: false,
		audio: false,
		video: false,
		reasoning: true,
		tools: true
	},
	{
		id: 'ggml-org/gemma-4-E4B-it-GGUF',
		downloads: 205894,
		likes: 105,
		lastModified: '2026-10-19T10:58:26.000Z',
		baseModel: 'google/gemma-4-E4B-it',
		license: 'apache-2.0',
		contextLength: 131072,
		params: 7518069290,
		architecture: 'gemma4',
		vision: true,
		audio: false,
		video: false,
		reasoning: true,
		tools: true
	},
	{
		id: 'ggml-org/Qwen3.8-27B-GGUF',
		downloads: 1412859,
		likes: 150,
		lastModified: '2026-10-19T15:35:52.000Z',
		baseModel: 'Qwen/Qwen3.8-27B',
		license: 'apache-2.0',
		contextLength: 262144,
		params: 26895998464,
		architecture: 'qwen35',
		vision: true,
		audio: false,
		video: false,
		reasoning: true,
		tools: true
	},
	{
		id: 'ggml-org/LFM2.5-24B-A2B-GGUF',
		downloads: 5194,
		likes: 4,
		lastModified: '2026-10-25T02:54:02.000Z',
		baseModel: 'LiquidAI/LFM2.5-24B-A2B',
		license: 'other',
		contextLength: 131072,
		params: 23971587712,
		architecture: 'lfm2moe',
		vision: false,
		audio: false,
		video: false,
		reasoning: true,
		tools: true
	},
	{
		id: 'ggml-org/Ministral-3-8B-Instruct-2512-GGUF',
		downloads: 636,
		likes: 1,
		lastModified: '2025-12-02T15:29:55.000Z',
		baseModel: 'mistralai/Ministral-3-8B-Instruct-2512-BF16',
		license: 'apache-2.0',
		contextLength: 262144,
		params: 8489553920,
		architecture: 'mistral3',
		vision: true,
		audio: false,
		video: false,
		reasoning: false,
		tools: true
	},
	{
		id: 'ggml-org/Devstral-Small-2-24B-Instruct-2512-GGUF',
		downloads: 4453,
		likes: 7,
		lastModified: '2025-12-18T15:46:07.000Z',
		baseModel: 'mistralai/Devstral-Small-2-24B-Instruct-2512',
		license: null,
		contextLength: 393216,
		params: 23572403200,
		architecture: 'mistral3',
		vision: true,
		audio: false,
		video: false,
		reasoning: false,
		tools: true
	}
];

/** Download counts the way Discover shows them, e.g. `1.4M`, `205.9K`. */
export function formatDownloads(downloads: number): string {
	if (downloads >= 1_000_000) return `${(downloads / 1_000_000).toFixed(1)}M`;

	if (downloads >= 1_000) return `${(downloads / 1_000).toFixed(1)}K`;

	return downloads.toString();
}

/** "3 weeks ago" from an ISO date, relative to the day the data was read. */
export function updatedAgo(iso: string, now = new Date('2026-11-07')): string {
	const days = Math.max(0, Math.floor((now.getTime() - new Date(iso).getTime()) / 86_400_000));

	if (days === 0) return 'today';

	if (days === 1) return 'yesterday';

	if (days < 7) return `${days} days ago`;

	if (days < 30) return `${Math.floor(days / 7)} weeks ago`;

	return `${Math.floor(days / 30)} months ago`;
}
