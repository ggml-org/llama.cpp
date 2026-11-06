// The social post and link preview templates: their formats, tones and the copy they ship with. The copy is the
// stories' args, so it can be rewritten in Storybook's controls before a post is exported.

/** neutral grounds take either lockup; the hero blue and navy take the Llama lockup only */
export type PostTone = 'dark' | 'hero' | 'light' | 'navy';

export interface PostFormat {
	height: number;
	name: string;
	width: number;
}

export const FORMATS = {
	landscape: { height: 900, name: 'Landscape 16:9', width: 1600 },
	og: { height: 630, name: 'Open Graph', width: 1200 },
	portrait: { height: 1350, name: 'Portrait 4:5', width: 1080 },
	repo: { height: 640, name: 'GitHub social preview', width: 1280 },
	square: { height: 1080, name: 'Square', width: 1080 },
	story: { height: 1920, name: 'Story 9:16', width: 1080 }
} satisfies Record<string, PostFormat>;

/** the lockup glyph, U+E002: the icon with the lockup's gap built in, typed straight before the name */
export const LOCKUP = '';
/** the icon on its own, U+E001: the capital form, for a header without the name */
export const ICON = '';

export interface Stat {
	label: string;
	value: string;
}

export interface Model {
	/** a short line over the name: the maker and what it is for */
	about?: string;
	/** the tag over the name, in cinder when `accent` */
	accent?: boolean;
	/** a Mono tag at the card's top right: its modality or kind */
	badge?: string;
	description?: string;
	/** Mono facts under the description: size, quant, licence */
	meta?: string[];
	name: string;
	repo?: string;
	stats?: Stat[];
	tag?: string;
}

export interface ModelDropCopy {
	chips: string[];
	headline: string;
	kicker: string;
	lede: string;
	models: Model[];
	section: string;
	url: string;
}

export const modelDrop: ModelDropCopy = {
	chips: ['route a ticket', 'moderate content', 'pick an agent’s next step'],
	headline: 'Decision models, running locally, a click away.',
	kicker: 'From the llama.cpp team & Hugging Face',
	lede: 'Give a model a document and typed questions. Get back a calibrated probability for every option, lightning fast.',
	models: [
		{
			accent: true,
			description:
				'From vLLM-SR’s Decision 3.0 family, #1 on the Jev Decision Index. Reads text, JSON, images and video.',
			meta: ['9B', '6.6 GB Q4_K_M', 'Apache-2.0'],
			name: 'd3-flash',
			repo: 'ggml-org/vllm-d3-flash-GGUF',
			tag: 'Jev Decision Index leader'
		},
		{
			description:
				'Liquid AI’s omni decision model. Text, images, audio and video, even spoken questions, in 2B.',
			meta: ['2B', '1.85 GB Q8_0', 'runs on a laptop'],
			name: 'd1-Omni-2B',
			repo: 'LiquidAI/d1-Omni-2B-GGUF',
			tag: 'Every modality'
		}
	],
	section: 'Fresh drops',
	url: 'llama.app/models'
};

export interface WhatsNewCopy {
	family: { about: string; aside: string; models: Model[]; name: string };
	footer: [string, string];
	headline: string;
	lede: string;
	models: Model[];
	pill: string;
	url: string;
}

export const whatsNew: WhatsNewCopy = {
	family: {
		about: 'Liquid AI · small and multimodal',
		aside: 'Built for on-device',
		models: [
			{
				badge: 'text',
				description: 'Small chat model in 16 languages. Runs on CPU or GPU.',
				name: 'LFM3-1.7B'
			},
			{
				badge: 'omni',
				description: 'A small model that can see, watch, hear and talk back.',
				name: 'LFM3-Omni-2B'
			},
			{
				badge: 'audio',
				description: 'Takes text and audio, outputs text or speech.',
				name: 'LFM3-Audio-2B'
			},
			{
				badge: 'decision',
				description: 'Yes/no, choice and score answers over text, image and video.',
				name: 'd1-2B'
			}
		],
		name: 'LFM3'
	},
	footer: ['By the llama.cpp team and Hugging Face', 'Free · Open source · Works offline'],
	headline: 'New in Llama.\nOn your computer.',
	lede: 'Liquid AI’s LFM3 family, Mistral Large 4 and Recursive’s Flint now run in the Llama app. Pick one from Models and start chatting.',
	models: [
		{
			about: 'Mistral AI · frontier scale',
			description:
				'The 2-bit quant fits in 512 GB of unified memory. Strong in cybersecurity and agentic coding.',
			name: 'Mistral Large 4',
			repo: 'ggml-org/Mistral-Large-4-1T-A52B-GGUF',
			stats: [
				{ label: 'params, ~52B active', value: '1T' },
				{ label: 'context, image input', value: '1M' }
			]
		},
		{
			about: 'Recursive · for coding agents',
			description:
				'Investigates codebases, fixes bugs and writes tests, with reasoning and tool use.',
			name: 'Flint',
			repo: 'ggml-org/Flint-GGUF',
			stats: [
				{ label: 'active of 39.8B', value: '2.5B' },
				{ label: 'SWE-bench Verified', value: '74.2%' }
			]
		}
	],
	pill: 'New models available today',
	url: 'llama.app'
};

export interface LinkPreviewCopy {
	caption: string;
	command: string;
	headline: string;
	repoHeadline: string;
	repoLede: string;
}

export const linkPreview: LinkPreviewCopy = {
	caption: 'Llama · Your AI, on your computer',
	command: 'llama-server -hf ggml-org/gemma-4-E4B-it-GGUF',
	headline: 'Your AI.\nOn your computer.',
	repoHeadline: 'LLM inference\nin C/C++',
	repoLede: 'Run open models on your own hardware, from a laptop to a server.'
};
