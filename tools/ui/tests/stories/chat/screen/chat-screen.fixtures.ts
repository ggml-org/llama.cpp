import { MODEL_NAME, modelServerProps, seedStores } from '../form/fixtures';
import { MessageRole, MessageType, ServerRole } from '$lib/enums';
import { DatabaseService } from '$lib/services/database.service';
import { chatStore, conversationsStore } from '$lib/stores';

export const SCREEN_CONVERSATION_ID = 'story-chat-screen';

export const REPLY_TEXT =
	'Sure. Quantization stores the weights with fewer bits, so the model takes less memory and runs faster, at a small cost in quality.';

const minutesAgo = (minutes: number) => Date.now() - minutes * 60_000;
const turn = (
	id: string,
	parent: string | null,
	children: string[],
	fields: Partial<DatabaseMessage>
): DatabaseMessage => ({
	children,
	content: '',
	convId: SCREEN_CONVERSATION_ID,
	id,
	parent,
	role: MessageRole.USER,
	timestamp: minutesAgo(10),
	toolCalls: '',
	type: MessageType.TEXT,
	...fields
});

/** A finished conversation in the export format, written to IndexedDB as the app would import it. */
export const screenConversation: ExportedConversation = {
	conv: {
		currNode: 'a2',
		id: SCREEN_CONVERSATION_ID,
		lastModified: minutesAgo(2),
		name: 'Running GGUF models locally'
	},
	messages: [
		turn('root', null, ['u1'], { role: MessageRole.SYSTEM, type: MessageType.ROOT }),
		turn('u1', 'root', ['a1'], {
			content: 'What is the difference between Q4_K_M and Q8_0?',
			timestamp: minutesAgo(9)
		}),
		turn('a1', 'u1', ['u2'], {
			content:
				'Both are GGUF quantization types:\n\n| Type | Bits per weight | Quality |\n| --- | --- | --- |\n| Q4_K_M | ~4.8 | good |\n| Q8_0 | ~8.5 | near lossless |\n\nPick **Q4_K_M** when memory is tight and **Q8_0** when quality matters most.',
			model: MODEL_NAME,
			reasoningContent:
				'The user compares two quant types. Give bits per weight and the trade-off, keep it short.',
			role: MessageRole.ASSISTANT,
			timestamp: minutesAgo(8),
			timings: { predicted_ms: 3100, predicted_n: 96, prompt_ms: 180, prompt_n: 42 }
		}),
		turn('u2', 'a1', ['a2'], {
			content: 'How do I run one with llama-server?',
			timestamp: minutesAgo(3)
		}),
		turn('a2', 'u2', [], {
			content:
				'Point the server at the file and open the web UI:\n\n```sh\nllama-server -m Qwen3-8B-Q4_K_M.gguf --port 8080\n```\n\nThen visit `http://localhost:8080`.',
			model: MODEL_NAME,
			role: MessageRole.ASSISTANT,
			timestamp: minutesAgo(2),
			timings: { cache_n: 120, predicted_ms: 2400, predicted_n: 64, prompt_ms: 90, prompt_n: 30 }
		})
	]
};

export interface FakeServerOptions {
	/** Text streamed back for a chat completion. */
	reply?: string;
	/** Delay between streamed chunks, in ms. */
	chunkDelayMs?: number;
	/** HTTP status for the completion request; anything but 200 is an error reply. */
	status?: number;
	/** Error message sent with a failed completion. */
	errorMessage?: string;
}

const sleep = (ms: number) => new Promise((resolve) => setTimeout(resolve, ms));
const json = (body: unknown, status = 200) =>
	new Response(JSON.stringify(body), { headers: { 'Content-Type': 'application/json' }, status });

function streamReply(reply: string, chunkDelayMs: number, signal?: AbortSignal | null): Response {
	const encoder = new TextEncoder();
	const words = reply.split(/(?<= )/);
	const sse = (data: unknown) => encoder.encode(`data: ${JSON.stringify(data)}\n\n`);
	const body = new ReadableStream<Uint8Array>({
		async start(controller) {
			for (const word of words) {
				if (signal?.aborted) return controller.error(new DOMException('Aborted', 'AbortError'));

				controller.enqueue(
					sse({ choices: [{ delta: { content: word }, index: 0 }], model: MODEL_NAME })
				);
				await sleep(chunkDelayMs);
			}

			controller.enqueue(
				sse({
					choices: [{ delta: {}, finish_reason: 'stop', index: 0 }],
					model: MODEL_NAME,
					timings: {
						predicted_ms: words.length * chunkDelayMs,
						predicted_n: words.length,
						prompt_ms: 40,
						prompt_n: 24
					}
				})
			);
			controller.enqueue(encoder.encode('data: [DONE]\n\n'));
			controller.close();
		}
	});

	return new Response(body, { headers: { 'Content-Type': 'text/event-stream' } });
}

/**
 * Stand-in for llama-server: answers chat completions with a streamed reply and
 * every other API call with a harmless empty result. Returns the restore.
 */
export function installFakeServer(options: FakeServerOptions = {}): () => void {
	const { chunkDelayMs = 30, errorMessage = 'Internal server error', reply = REPLY_TEXT } = options;
	const status = options.status ?? 200;
	const realFetch = window.fetch;

	window.fetch = async (input, init) => {
		const url = typeof input === 'string' ? input : input instanceof URL ? input.href : input.url;

		if (url.includes('/v1/chat/completions') && !url.includes('/control')) {
			if (status !== 200) return json({ error: { code: status, message: errorMessage } }, status);

			const body = JSON.parse(String(init?.body ?? '{}'));

			if (!body.stream) {
				return json({
					choices: [{ message: { content: reply, role: 'assistant' } }],
					timings: { cache_n: 0, prompt_n: 24 }
				});
			}

			return streamReply(reply, chunkDelayMs, init?.signal);
		}

		if (url.includes('/props')) return json(modelServerProps());

		if (url.includes('/v1/models')) return json({ data: [], object: 'list' });

		return json({}, 404);
	};

	return () => {
		window.fetch = realFetch;
	};
}

/** Single-model server state plus an empty new-chat screen. */
export function seedNewChat(): () => void {
	const restore = seedStores({
		activeConversation: null,
		activeMessages: [],
		config: { conversationTabs: false, showMessageStats: true },
		serverProps: modelServerProps(),
		serverRole: ServerRole.MODEL
	});

	chatStore.dismissErrorDialog();

	return restore;
}

/** Writes the fixture conversation to IndexedDB and opens it like the chat route does. */
export async function openScreenConversation(): Promise<void> {
	await DatabaseService.deleteConversation(SCREEN_CONVERSATION_ID).catch(() => {});
	await DatabaseService.importConversations([structuredClone(screenConversation)]);
	await conversationsStore.loadConversations();
	await conversationsStore.loadConversation(SCREEN_CONVERSATION_ID);
}
