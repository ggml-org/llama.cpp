import {
	AgenticSectionType,
	AttachmentType,
	BuiltInTool,
	MessageRole,
	MessageType
} from '$lib/enums';
import { chatStore, conversationsStore, settingsStore } from '$lib/stores';
import { chatActivityStore } from '$lib/stores/chat/activity.svelte';
import type {
	AgenticSection,
	ApiChatCompletionToolCall,
	ApiProcessingState,
	ChatMessageActions,
	ChatMessageAgenticTimings,
	ChatMessageSiblingInfo,
	DatabaseMessage,
	DatabaseMessageExtra,
	SettingsConfigValue
} from '$lib/types';

export const CONV_ID = 'conv-1';
export const MODEL_NAME = 'ggml-org/gpt-oss-20b-GGUF';
export const STORY_CLASS = 'max-w-[56rem] w-[calc(100vw-2rem)]';

const NOW = Date.now();

export const noopChatActions: ChatMessageActions = {
	continueAssistantMessage: () => {},
	copy: () => {},
	delete: () => {},
	editUserMessagePreserveResponses: () => {},
	editWithBranching: () => {},
	editWithReplacement: () => {},
	forkConversation: () => {},
	navigateToSibling: () => {},
	regenerateCompaction: () => {},
	regenerateWithBranching: () => {}
};

function createMessage(
	fields: Pick<DatabaseMessage, 'id' | 'role' | 'content'> & Partial<DatabaseMessage>
): DatabaseMessage {
	return {
		children: [],
		convId: CONV_ID,
		parent: null,
		timestamp: NOW - 1000 * 60,
		type: MessageType.TEXT,
		...fields
	};
}

function toolCallsJson(calls: { id: string; name: string; args: unknown }[]): string {
	const toolCalls: ApiChatCompletionToolCall[] = calls.map((call) => ({
		function: { arguments: JSON.stringify(call.args), name: call.name },
		id: call.id,
		type: 'function'
	}));

	return JSON.stringify(toolCalls);
}

// Small inline images so the stories need no binary assets.
const SVG_IMAGE_URL = `data:image/svg+xml;base64,${btoa(
	'<svg xmlns="http://www.w3.org/2000/svg" width="320" height="200" viewBox="0 0 320 200"><defs><linearGradient id="g" x1="0" y1="0" x2="1" y2="1"><stop offset="0" stop-color="#f59e0b"/><stop offset="1" stop-color="#ef4444"/></linearGradient></defs><rect width="320" height="200" fill="url(#g)"/><circle cx="160" cy="100" r="48" fill="#fff" fill-opacity="0.85"/></svg>'
)}`;
const CHART_IMAGE_URL = `data:image/svg+xml;base64,${btoa(
	'<svg xmlns="http://www.w3.org/2000/svg" width="240" height="160" viewBox="0 0 240 160"><rect width="240" height="160" fill="#0f172a"/><rect x="30" y="90" width="30" height="50" fill="#38bdf8"/><rect x="80" y="60" width="30" height="80" fill="#38bdf8"/><rect x="130" y="30" width="30" height="110" fill="#38bdf8"/><rect x="180" y="70" width="30" height="70" fill="#38bdf8"/></svg>'
)}`;

/* User messages */

export const userMessage = createMessage({
	content: 'What is the meaning of life, the universe, and everything?',
	id: 'user-1',
	role: MessageRole.USER
});

export const userMarkdownMessage = createMessage({
	content:
		'Can you review this function?\n\n```ts\nexport function sum(values: number[]): number {\n\treturn values.reduce((acc, v) => acc + v, 0);\n}\n```\n\n- Is it safe for empty arrays?\n- Should it accept `bigint`?',
	id: 'user-markdown',
	role: MessageRole.USER
});

export const userAttachments: DatabaseMessageExtra[] = [
	{
		base64Url: SVG_IMAGE_URL,
		name: 'sunset.svg',
		size: 412,
		type: AttachmentType.IMAGE
	},
	{
		content: 'id,name,score\n1,Ada,98\n2,Linus,91\n3,Grace,95\n',
		name: 'scores.csv',
		size: 46,
		type: AttachmentType.TEXT
	},
	{
		base64Data: btoa('%PDF-1.4\n%mock\n'),
		content: 'Quarterly report. Revenue grew 12 percent quarter over quarter.',
		name: 'report.pdf',
		processedAsImages: false,
		size: 18_432,
		type: AttachmentType.PDF
	}
];

export const userWithAttachmentsMessage = createMessage({
	content: 'Summarize the attached files and describe the image.',
	extra: userAttachments,
	id: 'user-attachments',
	role: MessageRole.USER
});

export const userAttachmentsOnlyMessage = createMessage({
	content: '',
	extra: [userAttachments[0]],
	id: 'user-attachments-only',
	role: MessageRole.USER
});

/* System messages */

export const systemMessage = createMessage({
	content: 'You are a helpful assistant. Answer briefly and use **markdown** when it helps.',
	id: 'system-1',
	role: MessageRole.SYSTEM,
	type: MessageType.SYSTEM
});

export const systemLongMessage = createMessage({
	content: [
		'You are a senior software engineer helping with the llama.cpp web UI.',
		'',
		'## Rules',
		'',
		'1. Read the relevant files before you answer.',
		'2. Keep answers short and concrete.',
		'3. Prefer existing helpers over new abstractions.',
		'4. Use ASCII only in code comments.',
		'5. Never invent API fields; read the types first.',
		'6. Point to file paths when you mention code.',
		'',
		'## Style',
		'',
		'- Use simple words.',
		'- Avoid long comments.',
		'- Show code only when it is needed.'
	].join('\n'),
	id: 'system-long',
	role: MessageRole.SYSTEM,
	type: MessageType.SYSTEM
});

/* Assistant messages */

export const assistantMessage = createMessage({
	content:
		'The answer to the ultimate question of life, the universe, and everything is **42**.\n\nThis comes from Douglas Adams\' "The Hitchhiker\'s Guide to the Galaxy," where a supercomputer named Deep Thought calculated this answer over 7.5 million years. However, the question itself was never properly formulated, which is why the answer seems meaningless without context.',
	id: 'assistant-1',
	parent: 'user-1',
	role: MessageRole.ASSISTANT
});

export const REASONING_TEXT =
	"Let's consider the question step by step:\n\n1. Identify the core problem\n2. Evaluate relevant information\n3. Formulate a clear answer\n\nFollowing this process keeps the final response focused and accurate.";

export const assistantWithReasoningMessage = createMessage({
	content: "Here's the concise answer, now that I've thought it through: **42**.",
	id: 'assistant-reasoning',
	parent: 'user-1',
	reasoningContent: REASONING_TEXT,
	role: MessageRole.ASSISTANT
});

// Reasoning with no answer after it, on a finished turn: the block shows "Cancelled".
export const assistantInterruptedReasoningMessage = createMessage({
	content: '',
	id: 'assistant-reasoning-interrupted',
	parent: 'user-1',
	reasoningContent: 'The user asks about the meaning of life. I should first check which',
	role: MessageRole.ASSISTANT
});

export const assistantWithStatsMessage = createMessage({
	content: assistantMessage.content,
	id: 'assistant-stats',
	model: MODEL_NAME,
	parent: 'user-1',
	role: MessageRole.ASSISTANT,
	timings: {
		cache_n: 128,
		predicted_ms: 3120,
		predicted_n: 164,
		prompt_ms: 412,
		prompt_n: 1830
	}
});

export const rawOutputMessage = createMessage({
	content: 'Hello there! How can I help you today?',
	id: 'assistant-raw',
	parent: 'user-1',
	reasoningContent:
		'User greeted me. Is this a trap? No, just a normal hello. Respond calmly and do not start explaining quantum physics again.',
	role: MessageRole.ASSISTANT
});

/* Agentic (tool calls) */

export const agenticTimings: ChatMessageAgenticTimings = {
	llm: { predicted_ms: 5400, predicted_n: 312, prompt_ms: 980, prompt_n: 4210 },
	perTurn: [
		{
			llm: { predicted_ms: 1800, predicted_n: 96, prompt_ms: 610, prompt_n: 2400 },
			toolCalls: [
				{ duration_ms: 42, name: BuiltInTool.SERVER_GET_INFO, success: true },
				{ duration_ms: 318, name: BuiltInTool.SERVER_EXEC_SHELL_COMMAND, success: true }
			],
			toolsMs: 360,
			turn: 1
		},
		{
			llm: { predicted_ms: 3600, predicted_n: 216, prompt_ms: 370, prompt_n: 1810 },
			toolCalls: [],
			toolsMs: 0,
			turn: 2
		}
	],
	toolCalls: [
		{ duration_ms: 42, name: BuiltInTool.SERVER_GET_INFO, success: true },
		{ duration_ms: 318, name: BuiltInTool.SERVER_EXEC_SHELL_COMMAND, success: true }
	],
	toolCallsCount: 2,
	toolsMs: 360,
	turns: 2
};

export const assistantWithToolCallsMessage = createMessage({
	content: 'Let me check the environment first.',
	id: 'assistant-tools',
	parent: 'user-1',
	reasoningContent: 'I need the working directory and the file list before I answer.',
	role: MessageRole.ASSISTANT,
	toolCalls: toolCallsJson([
		{ args: {}, id: 'call-info', name: BuiltInTool.SERVER_GET_INFO },
		{ args: { command: 'ls -la' }, id: 'call-ls', name: BuiltInTool.SERVER_EXEC_SHELL_COMMAND }
	])
});

export const toolResultMessages: DatabaseMessage[] = [
	createMessage({
		content: JSON.stringify({ cwd: '/home/user/projects/llama.cpp', os: 'linux' }),
		id: 'tool-info',
		parent: 'assistant-tools',
		role: MessageRole.TOOL,
		toolCallId: 'call-info'
	}),
	createMessage({
		content:
			'total 24\ndrwxr-xr-x  6 user user 4096 Oct 28 10:12 .\ndrwxr-xr-x 12 user user 4096 Oct 27 09:01 ..\n-rw-r--r--  1 user user 1067 Oct 20 14:33 LICENSE\n-rw-r--r--  1 user user 2311 Oct 28 10:12 README.md\ndrwxr-xr-x  4 user user 4096 Oct 28 10:10 src\ndrwxr-xr-x  3 user user 4096 Oct 26 18:40 tools\n[exit code: 0]',
		id: 'tool-ls',
		parent: 'tool-info',
		role: MessageRole.TOOL,
		toolCallId: 'call-ls',
		toolCwd: '/home/user/projects/llama.cpp'
	})
];

// Continuation turn after the tool results; it closes the agentic session.
export const agenticFinalTurnMessage = createMessage({
	content:
		'The project root holds `README.md`, `LICENSE` and two folders: `src` and `tools`. The shell runs on Linux in `~/projects/llama.cpp`.',
	id: 'assistant-tools-final',
	parent: 'tool-ls',
	role: MessageRole.ASSISTANT
});

export const agenticToolMessages: DatabaseMessage[] = [
	...toolResultMessages,
	agenticFinalTurnMessage
];

export const assistantAgenticWithStatsMessage: DatabaseMessage = {
	...assistantWithToolCallsMessage,
	id: 'assistant-tools-stats',
	model: MODEL_NAME,
	timings: {
		agentic: agenticTimings,
		predicted_ms: 1800,
		predicted_n: 96,
		prompt_ms: 610,
		prompt_n: 2400
	}
};

/* Compaction */

export const preCompactionAssistantMessage = createMessage({
	content: 'Deep Thought took 7.5 million years to answer.',
	id: 'assistant-pre-compaction',
	parent: 'user-1',
	role: MessageRole.ASSISTANT,
	timestamp: NOW - 1000 * 60 * 31,
	timings: { cache_n: 0, predicted_n: 118, prompt_n: 42200 }
});

export const compactionMessage = createMessage({
	content:
		'The user asked about the ultimate answer and its origin. Key facts: the answer is **42**, from Douglas Adams\' "The Hitchhiker\'s Guide to the Galaxy", computed by Deep Thought over 7.5 million years.\n\nNothing is left to do unless the user wants the question itself discussed.',
	id: 'compaction-1',
	model: MODEL_NAME,
	parent: 'assistant-pre-compaction',
	role: MessageRole.USER,
	timestamp: NOW - 1000 * 60 * 30,
	// The pre-encode stamps the reduced prompt; `compaction` keeps the summary request timings.
	timings: {
		cache_n: 0,
		compaction: {
			cache_n: 0,
			predicted_ms: 12000,
			predicted_n: 300,
			prompt_ms: 18200,
			prompt_n: 42200
		},
		predicted_n: 0,
		prompt_n: 3204
	},
	type: MessageType.COMPACTION
});

export const compactionPendingMessage = createMessage({
	content: '',
	id: 'compaction-pending',
	parent: 'assistant-pre-compaction',
	role: MessageRole.USER,
	type: MessageType.COMPACTION
});

export const compactionGeneratingMessage = createMessage({
	content:
		'The user asked about the ultimate answer and its origin. Key facts: the answer is **42**',
	id: 'compaction-generating',
	model: MODEL_NAME,
	parent: 'assistant-pre-compaction',
	role: MessageRole.USER,
	type: MessageType.COMPACTION
});

/* Synthetic and MCP prompt */

export const cwdChangeMessage = createMessage({
	content: 'Set working directory to [file:///home/user/projects/llama.cpp](~/projects/llama.cpp)',
	id: 'cwd-change',
	isSynthetic: true,
	role: MessageRole.USER
});

export const cwdClearedMessage = createMessage({
	content: 'Working directory cleared',
	id: 'cwd-cleared',
	isSynthetic: true,
	role: MessageRole.USER
});

export const mcpPromptMessage = createMessage({
	content: '',
	extra: [
		{
			arguments: { language: 'TypeScript', path: 'src/lib/utils/agentic.ts' },
			content:
				'Review the TypeScript file at src/lib/utils/agentic.ts. Point out bugs first, then style issues.',
			name: 'code_review',
			promptName: 'code_review',
			serverName: 'github',
			type: AttachmentType.MCP_PROMPT
		}
	],
	id: 'mcp-prompt',
	role: MessageRole.USER
});

/* Branching */

export const siblingInfo: ChatMessageSiblingInfo = {
	currentIndex: 1,
	message: assistantMessage,
	siblingIds: ['assistant-0', assistantMessage.id, 'assistant-2'],
	totalSiblings: 3
};

/* Tool call sections, as deriveAgenticSections builds them */

function toolSection(
	toolName: string,
	args: unknown,
	toolResult?: string,
	type: AgenticSectionType = AgenticSectionType.TOOL_CALL
): AgenticSection {
	return {
		content: toolResult ?? '',
		toolArgs: typeof args === 'string' ? args : JSON.stringify(args),
		toolCallId: `call-${toolName}`,
		toolName,
		toolResult,
		type
	};
}

export const defaultToolSection = toolSection(
	'get_weather',
	{ city: 'Warsaw', units: 'metric' },
	JSON.stringify({ city: 'Warsaw', condition: 'cloudy', humidity: 71, temperature: 9 })
);

export const defaultToolPendingSection = toolSection(
	'get_weather',
	{ city: 'Warsaw', units: 'metric' },
	undefined,
	AgenticSectionType.TOOL_CALL_PENDING
);

export const defaultToolStreamingSection = toolSection(
	'get_weather',
	'{"city": "Wars',
	undefined,
	AgenticSectionType.TOOL_CALL_STREAMING
);

export const defaultToolMarkdownSection = toolSection(
	'summarize_issue',
	{ issue: 31273 },
	'## Issue 31273\n\n**Status:** open\n\nThe reasoning menu should list the effort levels the chat template supports.'
);

export const getDatetimeSection = toolSection(
	BuiltInTool.BROWSER_GET_DATETIME,
	{},
	JSON.stringify({ result: 'Wednesday, October 28, 2026, 10:15:00 CET' })
);

export const getInfoSection = toolSection(
	BuiltInTool.SERVER_GET_INFO,
	{},
	JSON.stringify({ cwd: '/home/user/projects/llama.cpp', os: 'linux' })
);

export const readFileSection = toolSection(
	BuiltInTool.SERVER_READ_FILE,
	{ end_line: 9, path: 'src/lib/utils/sum.ts', start_line: 1 },
	'export function sum(values: number[]): number {\n\treturn values.reduce((acc, v) => acc + v, 0);\n}\n\nexport function mean(values: number[]): number {\n\tif (values.length === 0) return 0;\n\n\treturn sum(values) / values.length;\n}'
);

export const editFileSection = toolSection(
	BuiltInTool.SERVER_EDIT_FILE,
	{
		edits: [
			{
				new_text: '\tif (values.length === 0) return Number.NaN;',
				old_text: '\tif (values.length === 0) return 0;'
			}
		],
		path: 'src/lib/utils/sum.ts'
	},
	JSON.stringify({ edits_applied: 1, result: 'File edited' })
);

export const editFileErrorSection = toolSection(
	BuiltInTool.SERVER_EDIT_FILE,
	{
		edits: [{ new_text: 'const b = 2;', old_text: 'const a = 1;' }],
		path: 'src/lib/utils/sum.ts'
	},
	JSON.stringify({ error: 'old_text not found in file' })
);

export const writeFileSection = toolSection(
	BuiltInTool.SERVER_WRITE_FILE,
	{
		content: '# Notes\n\n- Sum is safe for empty arrays.\n- Mean returns NaN for empty input.\n',
		path: 'docs/notes.md'
	},
	JSON.stringify({ bytes: 79, result: 'File written' })
);

export const execShellSection: AgenticSection = {
	...toolSection(
		BuiltInTool.SERVER_EXEC_SHELL_COMMAND,
		{ command: 'npm test -- --run' },
		' RUN  v4.1.12 /home/user/projects/app\n\n  src/sum.test.ts (3 tests) 4ms\n  src/mean.test.ts (2 tests) 3ms\n\n Test Files  2 passed (2)\n      Tests  5 passed (5)\n[exit code: 0]'
	),
	toolCwd: '/home/user/projects/app'
};

export const execShellFailedSection: AgenticSection = {
	...toolSection(
		BuiltInTool.SERVER_EXEC_SHELL_COMMAND,
		{ command: 'cat missing.txt' },
		'cat: missing.txt: No such file or directory\n[exit code: 1]'
	),
	toolCwd: '/home/user/projects/app'
};

export const execShellTimedOutSection: AgenticSection = {
	...toolSection(
		BuiltInTool.SERVER_EXEC_SHELL_COMMAND,
		{ command: 'sleep 600' },
		'[exit code: -1] [exit due to timed out]'
	),
	toolCwd: '/home/user/projects/app'
};

export const execShellPendingSection: AgenticSection = toolSection(
	BuiltInTool.SERVER_EXEC_SHELL_COMMAND,
	{ command: 'npm run build' },
	undefined,
	AgenticSectionType.TOOL_CALL_PENDING
);

export const fileGlobSearchSection = toolSection(
	BuiltInTool.SERVER_FILE_GLOB_SEARCH,
	{ exclude: '**/node_modules/**', include: '**/*.svelte', path: 'src/lib/components' },
	JSON.stringify({
		plain_text_response:
			'src/lib/components/app/chat/ChatMessages/ChatMessage/ChatMessage.svelte\nsrc/lib/components/app/chat/ChatMessages/ChatMessageReasoningBlock.svelte\nsrc/lib/components/app/chat/ChatMessages/ChatMessageEditForm.svelte\n---\nTotal matches: 3'
	})
);

export const grepSearchSection = toolSection(
	BuiltInTool.SERVER_GREP_SEARCH,
	{
		include: '**/*.ts',
		path: 'src/lib',
		pattern: 'deriveAgenticSections',
		return_line_numbers: true
	},
	JSON.stringify({
		plain_text_response:
			'src/lib/utils/agentic.ts:209:export function deriveAgenticSections(\nsrc/lib/utils/index.ts:88:\tderiveAgenticSections,\n---\nTotal matches: 2'
	})
);

export const runJavascriptSection = toolSection(
	BuiltInTool.BROWSER_RUN_JAVASCRIPT,
	{
		code: 'const values = [3, 1, 2];\nconsole.log(values.sort());\nreturn values.length;',
		timeout_ms: 5000
	},
	'[1,2,3]\n3'
);

export const runJavascriptErrorSection = toolSection(
	BuiltInTool.BROWSER_RUN_JAVASCRIPT,
	{ code: 'undefinedFunction();' },
	JSON.stringify({ error: 'ReferenceError: undefinedFunction is not defined' })
);

export const readMediaSection: AgenticSection = {
	...toolSection(
		BuiltInTool.BROWSER_READ_MEDIA,
		{ path: '/home/user/charts/usage.svg' },
		'File: /home/user/charts/usage.svg\nSize: 512 bytes\nMIME: image/svg+xml\n[Attachment saved: mcp-attachment-usage.svg]'
	),
	toolResultExtras: [
		{
			base64Url: CHART_IMAGE_URL,
			name: 'mcp-attachment-usage.svg',
			type: AttachmentType.IMAGE
		}
	]
};

export const searchResultsSection = toolSection(
	'web_search_exa',
	{ query: 'llama.cpp speculative decoding' },
	[
		'Title: Speculative decoding in llama.cpp',
		'URL: https://github.com/ggml-org/llama.cpp/tree/master/examples/speculative',
		'Published: 2025-03-14',
		'Author: N/A',
		'Highlights:',
		'Use a small draft model to propose tokens that the main model verifies in one batch.',
		'---',
		'Title: llama-server README',
		'URL: https://github.com/ggml-org/llama.cpp/blob/master/tools/server/README.md',
		'Published: N/A',
		'Author: ggml-org',
		'Highlights:',
		'The server accepts --model-draft to enable speculative decoding.'
	].join('\n')
);

/* Reasoning sections */

export const reasoningSection: AgenticSection = {
	content: REASONING_TEXT,
	type: AgenticSectionType.REASONING
};

export const reasoningPendingSection: AgenticSection = {
	content: REASONING_TEXT,
	type: AgenticSectionType.REASONING_PENDING
};

export const reasoningInterruptedSection: AgenticSection = {
	content: 'The user asks about the meaning of life. I should first check which',
	type: AgenticSectionType.REASONING,
	wasInterrupted: true
};

/* Store seeding */

export const promptProcessingState: ApiProcessingState = {
	contextTotal: 32768,
	contextUsed: 10000,
	hasNextToken: false,
	outputTokensMax: -1,
	outputTokensUsed: 0,
	progressPercent: 42,
	promptProgress: { cache: 0, processed: 4200, time_ms: 6000, total: 10000 },
	promptTokens: 0,
	speculative: false,
	status: 'preparing',
	temperature: 0.8,
	tokensDecoded: 0,
	tokensPerSecond: 0,
	tokensRemaining: -1,
	topP: 0.95
};

export const generatingProcessingState: ApiProcessingState = {
	contextTotal: 32768,
	contextUsed: 45200,
	hasNextToken: true,
	outputTokensMax: -1,
	outputTokensUsed: 128,
	promptTokens: 42200,
	speculative: false,
	status: 'generating',
	temperature: 0.8,
	tokensDecoded: 128,
	tokensPerSecond: 25,
	tokensRemaining: -1,
	topP: 0.95
};

export interface ChatStorySeed {
	settings?: Record<string, SettingsConfigValue>;
	/** Branch the compaction toggle reads its "before" size from. */
	activeMessages?: DatabaseMessage[];
	/** Marks the conversation as having a local pipe (chatStore.isLoading). */
	loading?: boolean;
	/** Message id of a live stream (chatStore.isStreaming()). */
	streamingMessageId?: string;
	processing?: ApiProcessingState;
	compactingNodeId?: string;
	/** Opens the edit form of this message on mount. */
	pendingEditMessageId?: string;
}

// Pinned so stories look the same whatever the viewer has in localStorage.
const STORY_SETTINGS_DEFAULTS: Record<string, SettingsConfigValue> = {
	enableContinueGeneration: false,
	renderThinkingAsMarkdown: true,
	renderUserContentAsMarkdown: true,
	showAgenticTurnStats: false,
	showMessageStats: true,
	showRawOutputSwitch: false
};

/**
 * Seed the global stores a chat message reads. Use as a story `beforeEach`:
 * it runs before render and the returned cleanup resets the stores, so state
 * does not leak into the next story.
 */
export function seedChatStory(seed: ChatStorySeed = {}): () => void {
	const settings = { ...STORY_SETTINGS_DEFAULTS, ...seed.settings };
	const previousSettings = Object.fromEntries(
		Object.keys(settings).map((key) => [key, settingsStore.config[key]])
	);

	for (const [key, value] of Object.entries(settings)) {
		settingsStore.updateConfig(key, value);
	}

	conversationsStore.activeConversation = {
		currNode: null,
		id: CONV_ID,
		lastModified: NOW,
		name: 'Storybook conversation'
	};
	conversationsStore.activeMessages = seed.activeMessages ?? [];

	if (seed.loading) chatActivityStore.markLocal(CONV_ID);

	if (seed.streamingMessageId) {
		chatStore.chatStreamingStates.set(CONV_ID, {
			messageId: seed.streamingMessageId,
			response: ''
		});
	}

	if (seed.processing) {
		chatStore.processing.setActiveConversation(CONV_ID);
		chatStore.processing.setState(CONV_ID, seed.processing);
	}

	if (seed.compactingNodeId) chatActivityStore.markCompacting(CONV_ID, seed.compactingNodeId);

	if (seed.pendingEditMessageId) chatStore.pendingEditMessageId = seed.pendingEditMessageId;

	return () => {
		for (const [key, value] of Object.entries(previousSettings)) {
			settingsStore.updateConfig(key, value);
		}

		chatActivityStore.localEnded(CONV_ID);
		chatActivityStore.endCompacting(CONV_ID);
		chatStore.chatStreamingStates.delete(CONV_ID);
		chatStore.processing.setState(CONV_ID, null);
		chatStore.processing.setActiveConversation(null);
		chatStore.pendingEditMessageId = null;
		conversationsStore.activeMessages = [];
		conversationsStore.activeConversation = null;
	};
}
