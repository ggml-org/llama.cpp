import jpgAsset from '../../fixtures/assets/1.jpg?url';
import pdfAsset from '../../fixtures/assets/example.pdf?raw';
import svgAsset from '../../fixtures/assets/hf-logo.svg?url';
import {
	BuiltInTool,
	ChatFormCommandAction,
	FileMentionEntryType,
	MessageRole,
	MessageType,
	MimeTypeText,
	ReasoningEffort,
	ServerRole,
	SpecialFileType,
	ToolSource
} from '$lib/enums';
import { ToolsService } from '$lib/services/tools.service';
import {
	conversationsStore,
	mcpStore,
	modelsStore,
	serverStore,
	settingsStore,
	toolsStore
} from '$lib/stores';
import type {
	ChatFormCommand,
	DatabaseConversation,
	DatabaseMessage,
	FileMentionEntry,
	GlobEntry,
	MCPPromptInfo,
	MCPResourceInfo,
	MCPServerSettingsEntry,
	ServerToolInfo
} from '$lib/types';

export const FORM_CLASS = 'max-w-[56rem] w-[calc(100vw-2rem)]';

export const MODEL_PATH = '/models/Qwen3-8B-Q4_K_M.gguf';
export const MODEL_NAME = 'Qwen3-8B-Q4_K_M.gguf';
export const SERVER_HOME = '/home/user';
export const CONVERSATION_ID = 'story-conversation';

export { jpgAsset, svgAsset };

export const fileAttachments: ChatUploadedFile[] = [
	{
		file: new File([''], '1.jpg', { type: 'image/jpeg' }),
		id: '1',
		name: '1.jpg',
		preview: jpgAsset,
		size: 44891,
		type: 'image/jpeg'
	},
	{
		file: new File([''], 'hf-logo.svg', { type: 'image/svg+xml' }),
		id: '2',
		name: 'hf-logo.svg',
		preview: svgAsset,
		size: 1234,
		type: 'image/svg+xml'
	},
	{
		file: new File([pdfAsset], 'example.pdf', { type: 'application/pdf' }),
		id: '3',
		name: 'example.pdf',
		size: 351048,
		type: 'application/pdf'
	},
	{
		file: new File(['# Notes\n\nShort text file.'], 'notes.md', { type: MimeTypeText.MARKDOWN }),
		id: '4',
		name: 'notes.md',
		size: 26,
		textContent: '# Notes\n\nShort text file.',
		type: MimeTypeText.MARKDOWN
	}
];

export const loadingAttachment: ChatUploadedFile = {
	file: new File([], 'loading'),
	id: 'loading-prompt',
	isLoading: true,
	mcpPrompt: { promptName: 'code_review', serverName: 'github' },
	name: 'Code review',
	size: 0,
	type: SpecialFileType.MCP_PROMPT
};

export const mcpServers: MCPServerSettingsEntry[] = [
	{ enabled: true, id: 'github', name: 'GitHub', url: 'https://api.githubcopilot.com/mcp/' },
	{ enabled: true, id: 'docs', name: 'Docs', url: 'https://docs.example.com/mcp' }
];

export const mcpPrompts: MCPPromptInfo[] = [
	{
		arguments: [
			{ description: 'Repository in owner/name form', name: 'repo', required: true },
			{ description: 'Pull request number', name: 'pr', required: true },
			{ description: 'Focus area, e.g. security', name: 'focus' }
		],
		description: 'Review a pull request and list the main issues',
		name: 'code_review',
		serverName: 'github',
		title: 'Code review'
	},
	{
		description: 'Summarize open issues by label',
		name: 'issue_summary',
		serverName: 'github',
		title: 'Issue summary'
	},
	{
		arguments: [{ description: 'Topic to explain', name: 'topic', required: true }],
		description: 'Explain a topic from the docs',
		name: 'explain',
		serverName: 'docs',
		title: 'Explain'
	},
	{
		description: 'Write release notes from the changelog',
		name: 'release_notes',
		serverName: 'docs'
	}
];

export const mcpResources: MCPResourceInfo[] = [
	{
		mimeType: MimeTypeText.MARKDOWN,
		name: 'README.md',
		serverName: 'github',
		uri: 'repo://ggml-org/llama.cpp/README.md'
	},
	{
		mimeType: MimeTypeText.PLAIN,
		name: 'server-api.txt',
		serverName: 'docs',
		title: 'Server API',
		uri: 'docs://server/api'
	}
];

export const chatCommands: ChatFormCommand[] = [
	{
		action: ChatFormCommandAction.PROMPT,
		description: 'Insert an MCP prompt',
		disabled: false,
		name: 'prompt'
	},
	{
		action: ChatFormCommandAction.CWD,
		description: 'Set working directory',
		disabled: false,
		keywords: ['current working directory'],
		name: 'cwd'
	},
	{
		action: ChatFormCommandAction.MODEL,
		description: 'Select model',
		disabled: false,
		name: 'model'
	},
	{
		action: ChatFormCommandAction.COMPACT,
		description: 'Summarize the conversation to free up context',
		disabled: true,
		keywords: ['summarize', 'context'],
		name: 'compact'
	}
];

export const mentionEntries: FileMentionEntry[] = [
	{
		name: 'src',
		path: `${SERVER_HOME}/llama.cpp/tools/ui/src`,
		type: FileMentionEntryType.DIRECTORY
	},
	{
		name: 'ChatForm.svelte',
		path: `${SERVER_HOME}/llama.cpp/tools/ui/src/lib/components/app/chat/ChatForm/ChatForm.svelte`,
		type: FileMentionEntryType.FILE
	},
	{
		name: 'chat.d.ts',
		path: `${SERVER_HOME}/llama.cpp/tools/ui/src/lib/types/chat.d.ts`,
		type: FileMentionEntryType.FILE
	}
];

export const directoryResults = [
	`${SERVER_HOME}/llama.cpp`,
	`${SERVER_HOME}/llama.cpp/tools`,
	`${SERVER_HOME}/llama.cpp/tools/ui`,
	`${SERVER_HOME}/llama.cpp/tools/server`
];

/** Server tools that make the cwd chip and the mention search available. */
export const cwdServerTools: ServerToolInfo[] = [
	BuiltInTool.SERVER_FILE_GLOB_SEARCH,
	BuiltInTool.SERVER_READ_FILE
].map((tool) => ({
	definition: { function: { name: tool, parameters: {} }, type: 'function' },
	display_name: tool,
	permissions: { write: false },
	tool,
	type: ToolSource.SERVER,
	uses_cwd: true
}));

export function modelServerProps(
	overrides: Partial<ApiLlamaCppServerProps> = {}
): ApiLlamaCppServerProps {
	return {
		default_generation_settings: { n_ctx: 32768 },
		modalities: { audio: false, video: false, vision: true },
		model_path: MODEL_PATH,
		...overrides
	} as unknown as ApiLlamaCppServerProps;
}

export const routerModels: ModelOption[] = [
	{ capabilities: [], id: 'qwen3-8b', model: 'Qwen/Qwen3-8B-GGUF', name: 'Qwen3-8B' },
	{ capabilities: [], id: 'gemma-3-4b', model: 'google/gemma-3-4b-it-GGUF', name: 'gemma-3-4b-it' }
];

export const conversation: DatabaseConversation = {
	currNode: 'a2',
	id: CONVERSATION_ID,
	lastModified: Date.now(),
	name: 'Story conversation'
};

/** A short user/assistant exchange whose last turn used `promptTokens` of context. */
export function conversationMessages(promptTokens: number, outputTokens = 512): DatabaseMessage[] {
	const base = {
		children: [],
		convId: CONVERSATION_ID,
		timestamp: Date.now(),
		type: MessageType.TEXT
	};

	return [
		{ ...base, content: 'Explain the KV cache.', id: 'u1', parent: null, role: MessageRole.USER },
		{
			...base,
			content: 'The KV cache stores keys and values of past tokens.',
			id: 'a1',
			model: MODEL_NAME,
			parent: 'u1',
			role: MessageRole.ASSISTANT,
			timings: { predicted_ms: 4200, predicted_n: 380, prompt_ms: 120, prompt_n: 240 }
		},
		{ ...base, content: 'And how big can it get?', id: 'u2', parent: 'a1', role: MessageRole.USER },
		{
			...base,
			content: 'It grows with the context length.',
			id: 'a2',
			model: MODEL_NAME,
			parent: 'u2',
			role: MessageRole.ASSISTANT,
			timings: {
				cache_n: 620,
				predicted_ms: outputTokens * 11,
				predicted_n: outputTokens,
				prompt_ms: 300,
				prompt_n: promptTokens - 620
			}
		}
	] as DatabaseMessage[];
}

/** Store state a story needs; every field left out keeps the current value. */
export interface StoreSeed {
	activeConversation?: DatabaseConversation | null;
	activeMessages?: DatabaseMessage[];
	config?: Record<string, unknown>;
	globSearch?: (params: Record<string, unknown>) => Promise<Record<string, unknown>>;
	mcpPrompts?: MCPPromptInfo[];
	mcpResources?: MCPResourceInfo[];
	mcpServers?: MCPServerSettingsEntry[];
	modelProps?: Record<string, ApiLlamaCppServerProps>;
	models?: ModelOption[];
	pendingCwd?: string | null;
	reasoningEffort?: ReasoningEffort;
	routerModels?: ApiModelDataEntry[];
	selectedModelId?: string | null;
	selectedModelName?: string | null;
	serverHome?: string | null;
	serverProps?: ApiLlamaCppServerProps | null;
	serverRole?: ServerRole | null;
	serverTools?: ServerToolInfo[];
}

interface ToolsStoreInternals {
	_serverHome: string | null | undefined;
	_serverTools: ServerToolInfo['definition'][];
	cwdAwareTools: Set<string>;
}

interface PropsCache {
	cache: { delete(key: string): boolean; set(key: string, value: ApiLlamaCppServerProps): void };
	cacheVersion: number;
}

type Restore = () => void;

/** Override an object member; the restore puts back the original. */
export function override<T extends object, K extends keyof T>(
	target: T,
	key: K,
	value: T[K]
): Restore {
	const hadOwn = Object.prototype.hasOwnProperty.call(target, key);
	const previous = target[key];

	Object.defineProperty(target, key, { configurable: true, value, writable: true });

	return () => {
		if (hadOwn) target[key] = previous;
		else delete target[key];
	};
}

/**
 * Seed the global stores for one story. Nothing is persisted. Return the
 * result from `beforeEach` so Storybook restores the stores afterwards.
 */
export function seedStores(seed: StoreSeed): Restore {
	const restores: Restore[] = [];
	const tools = toolsStore as unknown as ToolsStoreInternals;
	const props = modelsStore.props as unknown as PropsCache;
	const set = <T extends object, K extends keyof T>(target: T, key: K, value: T[K]) => {
		const previous = target[key];

		target[key] = value;
		restores.push(() => (target[key] = previous));
	};

	if (seed.serverRole !== undefined) set(serverStore, 'role', seed.serverRole);

	if (seed.serverProps !== undefined) set(serverStore, 'props', seed.serverProps);

	if (seed.models) set(modelsStore, 'models', seed.models);

	if (seed.routerModels) set(modelsStore, 'routerModels', seed.routerModels);

	if (seed.selectedModelId !== undefined) set(modelsStore, 'selectedModelId', seed.selectedModelId);

	if (seed.selectedModelName !== undefined)
		set(modelsStore, 'selectedModelName', seed.selectedModelName);

	if (seed.config || seed.mcpServers) {
		set(settingsStore, 'config', {
			...settingsStore.config,
			...seed.config,
			...(seed.mcpServers ? { mcpServers: JSON.stringify(seed.mcpServers) } : {})
		});
	}

	if (seed.reasoningEffort !== undefined)
		set(conversationsStore.preferences, 'pendingReasoningEffort', seed.reasoningEffort);

	if (seed.pendingCwd !== undefined)
		set(conversationsStore.preferences, 'pendingCwd', seed.pendingCwd);

	if (seed.activeConversation !== undefined)
		set(conversationsStore, 'activeConversation', seed.activeConversation);

	if (seed.activeMessages) set(conversationsStore, 'activeMessages', seed.activeMessages);

	if (seed.serverTools) {
		set(
			tools,
			'_serverTools',
			seed.serverTools.map((t) => t.definition)
		);
		set(
			tools,
			'cwdAwareTools',
			new Set(seed.serverTools.filter((t) => t.uses_cwd).map((t) => t.tool))
		);
	}

	if (seed.serverHome !== undefined) set(tools, '_serverHome', seed.serverHome);

	if (seed.modelProps) {
		for (const [id, value] of Object.entries(seed.modelProps)) {
			props.cache.set(id, value);
			restores.push(() => props.cache.delete(id));
		}

		props.cacheVersion++;
	}

	if (seed.globSearch) {
		const globSearch = seed.globSearch;

		restores.push(
			override(ToolsService, 'executeToolRaw', async (toolName, params) => {
				if (toolName !== BuiltInTool.SERVER_FILE_GLOB_SEARCH) return {};

				return globSearch(params);
			})
		);
	}

	if (seed.mcpPrompts) {
		const prompts = seed.mcpPrompts;

		restores.push(override(mcpStore, 'ensureInitialized', async () => true));
		restores.push(override(mcpStore, 'getAllPrompts', async () => prompts));
	}

	if (seed.mcpResources) {
		for (const resource of seed.mcpResources) {
			const attachment = mcpStore.resources.addAttachment(resource);

			mcpStore.resources.updateAttachmentContent(attachment.id, [
				{ mimeType: resource.mimeType, text: `Content of ${resource.name}`, uri: resource.uri }
			]);
		}

		restores.push(() => mcpStore.resources.clearAttachments());
	}

	return () => {
		for (const restore of restores.reverse()) restore();
	};
}

/** A `file_glob_search` stand-in that returns `paths` relative to the server home. */
export function globSearchResult(entries: GlobEntry[], delayMs = 0) {
	return async () => {
		if (delayMs > 0) await new Promise((resolve) => setTimeout(resolve, delayMs));

		return { base: SERVER_HOME, entries };
	};
}

/** Never resolves: keeps a picker in its loading state. */
export function pendingGlobSearch() {
	return () => new Promise<Record<string, unknown>>(() => {});
}

export function toGlobEntries(paths: { path: string; type: FileMentionEntryType }[]): GlobEntry[] {
	return paths.map(({ path, type }) => ({
		path: path.replace(`${SERVER_HOME}/`, ''),
		type: type === FileMentionEntryType.DIRECTORY ? 'dir' : 'file'
	}));
}

export { ReasoningEffort, ServerRole };

/** An open menu that unmounts leaves `pointer-events: none` on the body; clear it. */
export function resetPointerEvents() {
	document.body.style.pointerEvents = '';
}
