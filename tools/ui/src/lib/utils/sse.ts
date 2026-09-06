import {
	SSE_DATA_PREFIX,
	SSE_DONE_MARKER,
	SSE_LINE_SEPARATOR,
	SSE_RECORD_SEPARATOR
} from '$lib/constants';

/**
 * Minimal SSE-with-JSON stream iterator.
 *
 * Yields one event per `\n\n`-separated record. Each event payload is the
 * decoded `data:` field after JSON-parsing. A `[DONE]` sentinel terminates
 * the stream early. Malformed records - any record whose `data:` payload
 * fails `JSON.parse` - are skipped silently: usually a transient mid-stream
 * fault that the caller should not have to special-case, and the noise of
 * logging every occurrence on long-running streams outweighs the diagnostic
 * value.
 *
 * Less ambitious than ChatService.handleStreamResponse (no resume, no byte
 * offset tracking) - suitable for one-shot streams like `/tools?stream=true`
 * where the consumer just reads chunks until done.
 */

export interface SseJsonEvent<T = unknown> {
	data: T;
}

/**
 * Splits a raw SSE byte buffer into complete records on the blank-line
 * boundary, returning the leftover partial record separately. Shared by the
 * record-based consumers (parseSseJsonStream, models.service).
 */
export function splitSseRecords(buffer: string): { records: string[]; rest: string } {
	const parts = buffer.split(SSE_RECORD_SEPARATOR);

	return { records: parts.slice(0, -1), rest: parts[parts.length - 1] ?? '' };
}

/**
 * Extracts the joined `data:` payload from one SSE record (the data lines
 * concatenated with a newline), or an empty string when the record carries
 * no data lines. Used by models.service to parse status envelopes.
 */
export function extractSseDataPayload(record: string): string {
	return record
		.split(SSE_LINE_SEPARATOR)
		.filter((line) => line.startsWith(SSE_DATA_PREFIX))
		.map((line) => line.slice(SSE_DATA_PREFIX.length).trim())
		.join(SSE_LINE_SEPARATOR);
}

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
			const { records, rest } = splitSseRecords(buffer);

			buffer = rest;

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
						// Skip silently per the function contract above.
					}
				}
			}
		}
	} finally {
		try {
			reader.releaseLock();
		} catch (error) {
			console.error('[sse] failed to release reader lock:', error);
		}
	}
}
