import { parseSseJsonStream } from '$lib/utils/sse';
import { describe, expect, it } from 'vitest';

function makeSseResponse(events: string[]): Response {
	const body = events.join('\n\n') + '\n\n';

	return new Response(body, {
		headers: { 'content-type': 'text/event-stream' },
		status: 200
	});
}

describe('parseSseJsonStream', () => {
	it('yields parsed data for each record', async () => {
		const response = makeSseResponse(['data: {"chunk": "a"}', 'data: {"chunk": "b"}']);
		const collected: unknown[] = [];

		for await (const ev of parseSseJsonStream(response)) {
			collected.push(ev.data);
		}
		expect(collected).toEqual([{ chunk: 'a' }, { chunk: 'b' }]);
	});

	it('stops on [DONE] sentinel', async () => {
		const response = makeSseResponse([
			'data: {"chunk": "a"}',
			'data: [DONE]',
			'data: {"chunk": "after-done"}'
		]);
		const collected: unknown[] = [];

		for await (const ev of parseSseJsonStream(response)) {
			collected.push(ev.data);
		}
		expect(collected).toEqual([{ chunk: 'a' }]);
	});

	it('skips malformed JSON records', async () => {
		const response = makeSseResponse([
			'data: {"chunk": "ok"}',
			'data: {not-json}',
			'data: {"chunk": "also-ok"}'
		]);
		const collected: unknown[] = [];

		for await (const ev of parseSseJsonStream(response)) {
			collected.push(ev.data);
		}
		expect(collected).toEqual([{ chunk: 'ok' }, { chunk: 'also-ok' }]);
	});

	it('handles records split across multiple chunks (partial last line)', async () => {
		const full = 'data: {"chunk": "x"}\n\ndata: {"chunk": "y"}\n\n';
		const stream = new ReadableStream<Uint8Array>({
			start(controller) {
				const enc = new TextEncoder();

				controller.enqueue(enc.encode(full.slice(0, full.length / 2)));
				controller.enqueue(enc.encode(full.slice(full.length / 2)));
				controller.close();
			}
		});
		const response = new Response(stream, {
			headers: { 'content-type': 'text/event-stream' },
			status: 200
		});
		const collected: unknown[] = [];

		for await (const ev of parseSseJsonStream(response)) {
			collected.push(ev.data);
		}
		expect(collected).toEqual([{ chunk: 'x' }, { chunk: 'y' }]);
	});

	it('returns immediately if response has no body', async () => {
		const response = new Response(null, { status: 200 });
		const collected: unknown[] = [];

		for await (const ev of parseSseJsonStream(response)) {
			collected.push(ev.data);
		}
		expect(collected).toEqual([]);
	});
});
