/**
 * Minimal SSE-with-JSON stream iterator.
 *
 * Yields one event per `\n\n`-separated record. Each event payload is the
 * decoded `data:` field after JSON-parsing. A `[DONE]` sentinel terminates
 * the stream early. Malformed records are skipped silently.
 *
 * Less ambitious than ChatService.handleStreamResponse (no resume, no byte
 * offset tracking) - suitable for one-shot streams like `/tools?stream=true`
 * where the consumer just reads chunks until done.
 */

export interface SseJsonEvent<T = unknown> {
	data: T;
}

const SSE_RECORD_SEPARATOR = '\n\n';
const SSE_LINE_SEPARATOR = '\n';
const SSE_DATA_PREFIX = 'data:';
const SSE_DONE_MARKER = '[DONE]';

export async function* parseSseJsonStream<T = unknown>(
	response: Response,
	signal?: AbortSignal
): AsyncGenerator<SseJsonEvent<T>> {
	const reader = response.body?.getReader();
	if (!reader) return;

	const decoder = new TextDecoder();
	let buffer = '';

	try {
		while (true) {
			if (signal?.aborted) return;

			const { done, value } = await reader.read();
			if (done) break;

			buffer += decoder.decode(value, { stream: true });
			const records = buffer.split(SSE_RECORD_SEPARATOR);
			buffer = records.pop() ?? '';

			for (const record of records) {
				if (!record) continue;
				for (const line of record.split(SSE_LINE_SEPARATOR)) {
					if (!line.startsWith(SSE_DATA_PREFIX)) continue;
					const payload = line.slice(SSE_DATA_PREFIX.length).trim();
					if (payload === SSE_DONE_MARKER) return;
					if (!payload) continue;
					try {
						yield { data: JSON.parse(payload) as T };
					} catch {
						// skip malformed lines
					}
				}
			}
		}
	} finally {
		try {
			reader.releaseLock();
		} catch {
			/* already released */
		}
	}
}
