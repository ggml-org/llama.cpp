// A few families from llama.app's catalog (ggml-org/llama-pages, src/lib/catalog.json), for the Scrapbook's
// llama.app proposal, with what the site reads from the Hugging Face Hub at build time (src/lib/catalog-hub.json,
// written by scripts/catalog-hub.js on 2026-11-07):
// - `org`: the maker's org, from the base model the first build quantizes, for ModelOrgAvatar;
// - `capabilities`: what a size can do, across its builds, worked out the way Llama Chat's Discover does.

export interface Capabilities {
	vision?: boolean;
	audio?: boolean;
	reasoning?: boolean;
	tools?: boolean;
}

export interface Build {
	quant?: string;
	sizeBytes?: number;
	memBytes?: number;
	repo: string;
}

export interface Size {
	name: string;
	params?: string;
	capabilities?: Capabilities;
	builds: Build[];
}

export interface Family {
	name: string;
	brand: string;
	org: string;
	description: string;
	released: string;
	license: string;
	nonCommercial?: boolean;
	featured?: boolean;
	decision?: boolean;
	embedding?: boolean;
	sizes: Size[];
}

export const FAMILIES: Family[] = [
	{
		name: 'GPT-OSS',
		brand: 'OpenAI',
		description:
			"OpenAI's open-weight family for **local reasoning and agents**, with tool use, function calling, and adjustable reasoning effort.",
		released: '2025-08',
		license: 'Apache 2.0',
		org: 'openai',
		featured: true,
		sizes: [
			{
				name: 'GPT-OSS 20B',
				params: '20B',
				capabilities: {
					reasoning: true,
					tools: true
				},
				builds: [
					{
						quant: 'mxfp4',
						sizeBytes: 12109566560,
						repo: 'ggml-org/gpt-oss-20b-GGUF'
					}
				]
			},
			{
				name: 'GPT-OSS 120B',
				params: '120B',
				capabilities: {
					reasoning: true,
					tools: true
				},
				builds: [
					{
						quant: 'mxfp4',
						sizeBytes: 63387346464,
						repo: 'ggml-org/gpt-oss-120b-GGUF'
					}
				]
			}
		]
	},
	{
		name: 'Gemma 4',
		brand: 'Gemma',
		description:
			"Google's most capable open **agentic models for your laptops**, with vision, audio, and support for 140+ languages.",
		released: '2026-03',
		license: 'Apache 2.0',
		org: 'google',
		featured: true,
		sizes: [
			{
				name: 'Gemma 4 E2B',
				params: 'E2B',
				capabilities: {
					vision: true,
					reasoning: true,
					tools: true
				},
				builds: [
					{
						quant: 'Q8_0',
						sizeBytes: 5524862368,
						repo: 'ggml-org/gemma-4-E2B-it-GGUF'
					}
				]
			},
			{
				name: 'Gemma 4 E4B',
				params: 'E4B',
				capabilities: {
					vision: true,
					reasoning: true,
					tools: true
				},
				builds: [
					{
						quant: 'Q8_0',
						sizeBytes: 8591114688,
						repo: 'ggml-org/gemma-4-E4B-it-GGUF'
					},
					{
						quant: 'Q4_0',
						sizeBytes: 4590807392,
						repo: 'ggml-org/gemma-4-E4B-it-GGUF'
					}
				]
			},
			{
				name: 'Gemma 4 12B',
				params: '12B',
				capabilities: {
					vision: true,
					reasoning: true,
					tools: true
				},
				builds: [
					{
						quant: 'Q8_0',
						sizeBytes: 12669645728,
						repo: 'ggml-org/gemma-4-12B-it-GGUF'
					},
					{
						quant: 'Q4_0',
						sizeBytes: 7219673216,
						repo: 'ggml-org/gemma-4-12B-it-GGUF'
					}
				]
			},
			{
				name: 'Gemma 4 26B-A4B',
				params: '26B-A4B',
				capabilities: {
					vision: true,
					reasoning: true,
					tools: true
				},
				builds: [
					{
						quant: 'Q8_0',
						sizeBytes: 27666266464,
						repo: 'ggml-org/gemma-4-26B-A4B-it-GGUF'
					},
					{
						quant: 'Q4_0',
						sizeBytes: 14618145824,
						repo: 'ggml-org/gemma-4-26B-A4B-it-GGUF'
					}
				]
			},
			{
				name: 'Gemma 4 31B',
				params: '31B',
				capabilities: {
					vision: true,
					reasoning: true,
					tools: true
				},
				builds: [
					{
						quant: 'Q8_0',
						sizeBytes: 33445215840,
						repo: 'ggml-org/gemma-4-31B-it-GGUF'
					},
					{
						quant: 'Q4_0',
						sizeBytes: 17992313088,
						repo: 'ggml-org/gemma-4-31B-it-GGUF'
					}
				]
			}
		]
	},
	{
		name: 'Qwen 3.8',
		brand: 'Qwen',
		description:
			"Alibaba's multimodal **mid-sized models for coding and research**, long-horizon agentic work with 256K native context.",
		released: '2026-08',
		license: 'Apache 2.0',
		org: 'Qwen',
		featured: true,
		sizes: [
			{
				name: 'Qwen 3.8 27B',
				params: '27B',
				capabilities: {
					vision: true,
					reasoning: true,
					tools: true
				},
				builds: [
					{
						quant: 'Q8_0',
						sizeBytes: 28595763552,
						repo: 'ggml-org/Qwen3.8-27B-GGUF'
					},
					{
						quant: 'Q4_K_M',
						sizeBytes: 18973870432,
						repo: 'ggml-org/Qwen3.8-27B-GGUF'
					}
				]
			}
		]
	},
	{
		name: 'Ministral 3',
		brand: 'Mistral',
		description:
			"Mistral AI's compact **multimodal models for on-device apps**, offering strong performance across 3B, 8B, and 14B sizes.",
		released: '2025-12',
		license: 'Apache 2.0',
		org: 'mistralai',
		sizes: [
			{
				name: 'Ministral 3 3B',
				params: '3B',
				capabilities: {
					vision: true,
					tools: true
				},
				builds: [
					{
						quant: 'Q8_0',
						sizeBytes: 4099964832,
						repo: 'ggml-org/Ministral-3-3B-Instruct-2512-GGUF'
					},
					{
						quant: 'Q4_K_M',
						sizeBytes: 2147023008,
						repo: 'mistralai/Ministral-3-3B-Instruct-2512-GGUF'
					}
				]
			},
			{
				name: 'Ministral 3 3B Reasoning',
				params: '3B Reasoning',
				capabilities: {
					vision: true,
					tools: true
				},
				builds: [
					{
						quant: 'Q8_0',
						sizeBytes: 4099963264,
						repo: 'ggml-org/Ministral-3-3B-Reasoning-2512-GGUF'
					},
					{
						quant: 'Q4_K_M',
						sizeBytes: 2147021472,
						repo: 'mistralai/Ministral-3-3B-Reasoning-2512-GGUF'
					}
				]
			},
			{
				name: 'Ministral 3 8B',
				params: '8B',
				capabilities: {
					vision: true,
					tools: true
				},
				builds: [
					{
						quant: 'Q8_0',
						sizeBytes: 9486069920,
						repo: 'ggml-org/Ministral-3-8B-Instruct-2512-GGUF'
					},
					{
						quant: 'Q4_K_M',
						sizeBytes: 5198911904,
						repo: 'mistralai/Ministral-3-8B-Instruct-2512-GGUF'
					}
				]
			},
			{
				name: 'Ministral 3 8B Reasoning',
				params: '8B Reasoning',
				capabilities: {
					vision: true,
					tools: true
				},
				builds: [
					{
						quant: 'Q8_0',
						sizeBytes: 9486068384,
						repo: 'ggml-org/Ministral-3-8B-Reasoning-2512-GGUF'
					},
					{
						quant: 'Q4_K_M',
						sizeBytes: 5198910368,
						repo: 'mistralai/Ministral-3-8B-Reasoning-2512-GGUF'
					}
				]
			},
			{
				name: 'Ministral 3 14B',
				params: '14B',
				capabilities: {
					vision: true,
					tools: true
				},
				builds: [
					{
						quant: 'Q8_0',
						sizeBytes: 14827658560,
						repo: 'ggml-org/Ministral-3-14B-Instruct-2512-GGUF'
					},
					{
						quant: 'Q4_K_M',
						sizeBytes: 8239593024,
						repo: 'mistralai/Ministral-3-14B-Instruct-2512-GGUF'
					}
				]
			},
			{
				name: 'Ministral 3 14B Reasoning',
				params: '14B Reasoning',
				capabilities: {
					vision: true,
					tools: true
				},
				builds: [
					{
						quant: 'Q8_0',
						sizeBytes: 14827657024,
						repo: 'ggml-org/Ministral-3-14B-Reasoning-2512-GGUF'
					},
					{
						quant: 'Q4_K_M',
						sizeBytes: 8239591488,
						repo: 'mistralai/Ministral-3-14B-Reasoning-2512-GGUF'
					}
				]
			}
		]
	},
	{
		name: 'LFM 2.5',
		brand: 'Liquid AI',
		description: "Liquid AI's **fast local model** for reasoning and tool use.",
		released: '2026-10',
		license: 'LFM Open License 1.0',
		org: 'LiquidAI',
		featured: true,
		sizes: [
			{
				name: 'LFM 2.5 24B-A2B',
				params: '24B-A2B',
				capabilities: {
					reasoning: true,
					tools: true
				},
				builds: [
					{
						quant: 'Q8_0',
						sizeBytes: 25493733408,
						repo: 'ggml-org/LFM2.5-24B-A2B-GGUF'
					},
					{
						quant: 'Q4_K_M',
						sizeBytes: 14526259232,
						repo: 'ggml-org/LFM2.5-24B-A2B-GGUF'
					}
				]
			}
		]
	},
	{
		name: 'Devstral 2',
		brand: 'Mistral',
		description:
			"Mistral AI's **mid-sized models for autonomous software agents**, built to explore repositories, edit across files, and solve engineering tasks.",
		released: '2026-01',
		license: 'Apache 2.0 / Modified MIT',
		org: 'mistralai',
		sizes: [
			{
				name: 'Devstral 2 24B',
				params: '24B',
				capabilities: {
					vision: true,
					tools: true
				},
				builds: [
					{
						quant: 'Q8_0',
						sizeBytes: 25055308352,
						repo: 'ggml-org/Devstral-Small-2-24B-Instruct-2512-GGUF'
					},
					{
						quant: 'Q4_K_M',
						sizeBytes: 14334446752,
						repo: 'unsloth/Devstral-Small-2-24B-Instruct-2512-GGUF'
					}
				]
			},
			{
				name: 'Devstral 2 123B',
				params: '123B',
				capabilities: {
					tools: true
				},
				builds: [
					{
						quant: 'Q8_0',
						sizeBytes: 132854938656,
						repo: 'ggml-org/Devstral-2-123B-Instruct-2512-GGUF'
					},
					{
						quant: 'Q4_K_M',
						sizeBytes: 74897662400,
						repo: 'unsloth/Devstral-2-123B-Instruct-2512-GGUF'
					}
				]
			}
		]
	},
	{
		name: 'Bespoke Nimble',
		brand: 'Bespoke Labs',
		description:
			"Bespoke Labs' **mid-sized decision model**. Given a context and typed questions, it returns a probability for each option.",
		released: '2026-09',
		license: 'CC BY-NC 4.0',
		org: 'bespokelabs',
		nonCommercial: true,
		decision: true,
		sizes: [
			{
				name: 'Bespoke Nimble 9B',
				params: '9B',
				capabilities: {
					reasoning: true,
					tools: true
				},
				builds: [
					{
						quant: 'Q8_0',
						sizeBytes: 9527503392,
						repo: 'ggml-org/Bespoke-Nimble-9B-v3-GGUF'
					},
					{
						quant: 'Q4_K_M',
						sizeBytes: 6324185632,
						repo: 'ggml-org/Bespoke-Nimble-9B-v3-GGUF'
					}
				]
			}
		]
	},
	{
		name: 'EmbeddingGemma 2',
		brand: 'Gemma',
		description:
			"Google's multimodal embedding model. Maps text, images, audio, and code into one shared vector space for search and retrieval.",
		released: '2026-10',
		license: 'Apache 2.0',
		org: 'google',
		embedding: true,
		sizes: [
			{
				name: 'EmbeddingGemma 2',
				params: '740M',
				capabilities: {
					vision: true
				},
				builds: [
					{
						quant: 'Q8_0',
						sizeBytes: 309855456,
						repo: 'ggml-org/embeddinggemma-2-GGUF'
					}
				]
			}
		]
	}
];
