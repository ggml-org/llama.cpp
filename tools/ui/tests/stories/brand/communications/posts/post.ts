// The social post and link preview templates: their formats, tones and the copy they ship with. The copy is the
// stories' args, so it can be rewritten in Storybook's controls before a post is exported.

/**
 * The grounds a piece is set on. The neutrals (light, dark) take either lockup; the day sky (llama.app's hero) and
 * the night sky, and sand take the Llama lockup only. Cinder is never a ground: it is the one accent on top of them
 */
export type Ground = 'dark' | 'day' | 'light' | 'night' | 'sand';
/** the older name, kept for the post templates */
export type PostTone = Ground;

export const GROUNDS: Record<Ground, { lockups: string; name: string; note: string }> = {
	dark: { lockups: 'Llama, llama.cpp', name: 'Dark', note: 'Ink, for the developer side' },
	day: {
		lockups: 'Llama',
		name: 'Day sky',
		note: "llama.app's hero: deeper at the top, light at the horizon"
	},
	light: { lockups: 'Llama, llama.cpp', name: 'Light', note: 'The default: neutral 50 with ink' },
	night: {
		lockups: 'Llama',
		name: 'Night sky',
		note: 'Darkest at the top, a faint glow at the horizon'
	},
	sand: { lockups: 'Llama', name: 'Sand', note: 'The warm neutral, with its own ink' }
};

export interface PostFormat {
	height: number;
	name: string;
	/** the pixels at the top and bottom the platform covers with its own bars */
	safe?: { bottom: number; top: number };
	/** where it goes */
	use?: string;
	width: number;
}

export const FORMATS = {
	billboard: { height: 800, name: 'Billboard 3:1', use: 'Banners, screens at events', width: 2400 },
	landscape: { height: 900, name: 'Landscape 16:9', use: 'X, Bluesky, slides', width: 1600 },
	/** the Mac App Store's screenshots: 16:10 at the largest size Apple takes, a 1440 × 900 screen at 2x */
	macScreenshot: {
		height: 1800,
		name: 'Mac App Store screenshot 16:10',
		use: 'Mac App Store screenshots, up to ten',
		width: 2880
	},
	/** the Microsoft Store's listing: artwork, screenshots and logos; the Store may lay text over the bottom quarter */
	msHero: {
		height: 1080,
		name: 'Super hero art 16:9',
		safe: { bottom: 270, top: 0 },
		use: 'The top of the Microsoft Store listing',
		width: 1920
	},
	msPoster: {
		height: 1080,
		name: 'Poster art 2:3',
		use: 'The Microsoft Store’s main logo',
		width: 720
	},
	msScreenshot: {
		height: 1080,
		name: 'Store screenshot 16:9',
		safe: { bottom: 270, top: 0 },
		use: 'Microsoft Store screenshots, four or more',
		width: 1920
	},
	msTile: { height: 300, name: 'App tile icon', use: 'The Microsoft Store’s tile', width: 300 },
	og: { height: 630, name: 'Open Graph', use: 'Link previews on llama.app', width: 1200 },
	portrait: { height: 1350, name: 'Portrait 4:5', use: 'Instagram, LinkedIn', width: 1080 },
	poster: { height: 1754, name: 'Poster A-series', use: 'Print, at 150 dpi', width: 1240 },
	repo: { height: 640, name: 'GitHub social preview', use: 'The repo’s link preview', width: 1280 },
	square: { height: 1080, name: 'Square', use: 'Instagram, LinkedIn, X', width: 1080 },
	story: {
		height: 1920,
		name: 'Story 9:16',
		safe: { bottom: 320, top: 220 },
		use: 'Instagram and TikTok stories, Shorts',
		width: 1080
	}
} satisfies Record<string, PostFormat>;

export type FormatKey = keyof typeof FORMATS;

export type Shape = 'square' | 'tall' | 'wide';

export interface Grid {
	/** the column count: six on square and tall formats, twelve on wide ones */
	columns: number;
	/** the space between columns, half a unit */
	gutter: number;
	/** the margin on every side, two units (plus the safe area) */
	margin: number;
	shape: Shape;
	/**
	 * the unit: a 24th of the short side, or a 48th of the long side on a format wider than 2:1. Every size in a piece
	 * (type, spacing, radii) is a multiple of it
	 */
	unit: number;
}

/**
 * The grid for a format: one unit tied to its short side, so a layout scales across formats rather than being redrawn.
 * `unit` fixes it instead, for a specimen tile sized to its content
 */
export function grid({ height, width }: PostFormat, unit?: number): Grid {
	unit ??= Math.max(Math.min(width, height) / 24, Math.max(width, height) / 48);
	const shape: Shape = width / height >= 1.5 ? 'wide' : height / width >= 1.3 ? 'tall' : 'square';

	return { columns: shape === 'wide' ? 12 : 6, gutter: unit / 2, margin: unit * 2, shape, unit };
}

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

export interface ReceiptCopy {
	footer: [string, string];
	headline: string;
	rows: [string, string][];
	title: string;
}

/** an early llama.app ad, said plainly: the headline states it, the receipt makes the point about paying per token */
export const openModels: ReceiptCopy = {
	footer: ['Free AI for everyone', 'Open source · Works offline'],
	headline: '100,000+ open models.\nOne computer: yours.',
	rows: [
		['tokens generated', '1,284,902'],
		['account', 'none'],
		['total', '$0.00']
	],
	title: 'Your AI bill, today'
};

export interface AgentsCopy {
	agents: { name: string; note: string }[];
	command: string;
	endpoint: string;
	footer: [string, string];
	headline: string;
	lede: string;
	url: string;
}

/**
 * Llama serving agents, after Merve's post: the endpoint any agent can point at, and how four of them connect. Her
 * lede's last sentence (personal agents connect during onboarding) is left to the Hermes card, which says it
 */
export const agentsPost: AgentsCopy = {
	agents: [
		{ name: 'OpenClaw', note: 'OpenAI-compatible API' },
		{ name: 'Hermes', note: 'Set up during onboarding' },
		{ name: 'Pi', note: '/login llama.cpp' },
		{ name: 'Codex', note: '~/.codex/config.toml' }
	],
	command: 'base_url = "http://127.0.0.1:9931/v1"',
	endpoint: ':9931/v1',
	footer: ['By the llama.cpp team and Hugging Face', 'Free · Open source · Works offline'],
	headline: 'Any agent.\nOn your computer.',
	lede: 'Llama serves open models at an OpenAI-compatible endpoint. Point your coding agent at it and keep working, even offline.',
	url: 'llama.app/docs/agents'
};

export interface StoreLine {
	headline: string;
	lede?: string;
}

/** a version of a screenshot's copy: one of several to compare, or to A/B test in a Store product page experiment */
export interface StoreVariant extends StoreLine {
	/** a short name for it, A, B and so on */
	id: string;
	/** what it leans on */
	note: string;
}

/**
 * The Microsoft Store listing, in the order the Store shows it (ten screenshots, its most): each one practical thing
 * the app does on Windows, said so that what we stand for comes through (yours, private, for everyone) without
 * preaching it, and only what the app already does (its release notes, its README). The opener and the chat come in
 * several versions, the opener's mostly leading with privacy; all in the voice: sentence case, full stops, no
 * superlatives
 */
export interface MicrosoftStoreCopy {
	apps: StoreLine;
	/** the chat, each version with its own conversation (a Store scene) */
	chats: (StoreVariant & { scene: 'chat' | 'document' | 'screenshot' })[];
	gpu: StoreLine;
	models: StoreLine;
	openers: StoreVariant[];
	privacy: StoreLine;
	quick: StoreLine;
	speed: StoreLine & { figureLabel: string; value: string };
	tray: StoreLine;
	verdicts: StoreLine;
}

export const microsoftStore: MicrosoftStoreCopy = {
	apps: {
		headline: 'Works with the apps\nyou already use.',
		lede: 'Any app that speaks the OpenAI API can use your models, right on your PC.'
	},
	chats: [
		{
			headline: 'Help with everyday things,\neven offline.',
			id: 'A',
			lede: 'Your conversations stay on your PC.',
			note: 'Writing',
			scene: 'chat'
		},
		{
			headline: 'Your documents\nstay on your PC.',
			id: 'B',
			lede: 'Summarize a lease, a bill or a letter, even offline.',
			note: 'Documents',
			scene: 'document'
		},
		{
			headline: 'Ask about what’s\non your screen.',
			id: 'C',
			lede: 'Share a screenshot for a plain answer. It never leaves your PC.',
			note: 'Screenshots',
			scene: 'screenshot'
		}
	],
	gpu: {
		headline: 'Runs on the graphics card\nyou already have.',
		lede: 'NVIDIA, AMD or Intel, or on the processor alone.'
	},
	models: {
		headline: 'The models you choose\nstay yours.',
		lede: 'Each one is a file on your PC: open its folder, run it or delete it. No one else can change it.'
	},
	openers: [
		{
			headline: 'Your AI.\nOn your PC.',
			id: 'A',
			lede: 'Free and open source. No account, no subscription.',
			note: 'Ownership'
		},
		{
			headline: 'Chat with AI,\nprivately.',
			id: 'B',
			lede: 'It runs on your PC. Your conversations stay with you.',
			note: 'Privacy, as what you do'
		},
		{
			headline: 'Run your own AI.\nLocal and private.',
			id: 'C',
			lede: 'On your PC, even offline. No account, no subscription.',
			note: 'Privacy, with ownership'
		},
		{
			headline: 'Local AI.\nFree and private.',
			id: 'D',
			lede: 'Run open models from Hugging Face on your PC.',
			note: 'Privacy, with price'
		},
		{
			headline: 'Private AI,\nright on your PC.',
			id: 'E',
			lede: 'Your conversations stay on it. Free, and it works offline.',
			note: 'Privacy, with place'
		}
	],
	privacy: {
		headline: 'No telemetry.\nNone.',
		lede: 'No events, no analytics, no install ID. Your prompts and replies stay on your PC.'
	},
	quick: {
		headline: 'A quick chat,\nfrom any app.',
		lede: 'Esc hides it, and your conversation waits for the next time.'
	},
	speed: {
		figureLabel: 'faster, with the same answers',
		headline: 'Faster replies,\nsame answers.',
		lede: 'Models that can draft ahead do it on their own. Nothing to set up.',
		value: 'Up to 2×'
	},
	tray: {
		headline: 'Good models to start with,\nright in the tray.',
		lede: 'Llama runs the server for you. Pick one, or search all of Hugging Face.'
	},
	verdicts: {
		headline: 'Know before\nyou download.',
		lede: 'Every version says whether it fits your PC, from what it needs and what your PC has.'
	}
};

/**
 * The Mac App Store listing: the same voice as the Microsoft Store's, for Llama for Mac and what it does that the
 * Windows app doesn't (its menu bar, a Safari web app in the Dock, its models reached from a phone), from its release
 * notes and README. It claims nothing about telemetry: the Mac app reports crashes
 */
export interface MacAppStoreCopy {
	apps: StoreLine;
	chats: (StoreVariant & { scene: 'chat' | 'document' })[];
	hardware: StoreLine;
	menu: StoreLine;
	openers: StoreVariant[];
	phone: StoreLine;
	size: StoreLine & { figureLabel: string; value: string };
}

export const macAppStore: MacAppStoreCopy = {
	apps: {
		headline: 'Works with the apps\nyou already use.',
		lede: 'Any app that speaks the OpenAI API can use your models, right on your Mac.'
	},
	chats: [
		{
			headline: 'Chat in a window\nof its own.',
			id: 'A',
			lede: 'Add Llama to the Dock as a Safari web app. Your conversations stay on your Mac.',
			note: 'Writing',
			scene: 'chat'
		},
		{
			headline: 'Your documents\nstay on your Mac.',
			id: 'B',
			lede: 'Summarize a lease, a bill or a letter, even offline.',
			note: 'Documents',
			scene: 'document'
		}
	],
	hardware: {
		headline: 'Runs on the Mac\nyou already have.',
		lede: 'Recommended for your Mac lists the models that fit its memory.'
	},
	menu: {
		headline: 'Lives in your\nmenu bar.',
		lede: 'Llama runs the server for you and suggests a first model that fits your Mac.'
	},
	openers: [
		{
			headline: 'Your AI.\nOn your Mac.',
			id: 'A',
			lede: 'Free and open source. No account, no subscription.',
			note: 'Ownership'
		},
		{
			headline: 'Chat with AI,\nprivately.',
			id: 'B',
			lede: 'It runs on your Mac. Your conversations stay with you.',
			note: 'Privacy, as what you do'
		},
		{
			headline: 'Run your own AI.\nLocal and private.',
			id: 'C',
			lede: 'On your Mac, even offline. No account, no subscription.',
			note: 'Privacy, with ownership'
		},
		{
			headline: 'Local AI.\nFree and private.',
			id: 'D',
			lede: 'Run open models from Hugging Face on your Mac.',
			note: 'Privacy, with price'
		},
		{
			headline: 'Private AI,\nright on your Mac.',
			id: 'E',
			lede: 'Your conversations stay on it. Free, and it works offline.',
			note: 'Privacy, with place'
		}
	],
	phone: {
		headline: 'Your models,\non your phone too.',
		lede: 'Scan the code in the menu. Over Tailscale, only your own devices can reach them.'
	},
	size: {
		figureLabel: 'a native Mac app',
		headline: 'Small, and\nmade for the Mac.',
		lede: 'It shares one llama.cpp with your terminal, and your models with your other tools.',
		value: '4 MB'
	}
};
