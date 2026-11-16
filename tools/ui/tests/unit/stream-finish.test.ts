import { ChatService } from '$lib/services/chat.service';
import { describe, expect, it } from 'vitest';

// a streamed answer of one text delta, the chunk that ends it saying why
function answer(finishReason: string): Response {
	const chunks = [
		{ choices: [{ delta: { content: 'part' }, finish_reason: null }] },
		{ choices: [{ delta: {}, finish_reason: finishReason }] }
	];
	const body = chunks.map((c) => `data: ${JSON.stringify(c)}\n\n`).join('') + 'data: [DONE]\n\n';

	return new Response(body, { headers: { 'content-type': 'text/event-stream' } });
}

describe('ChatService stream end', () => {
	it('tells why the model stopped, length once the output limit cut the answer', async () => {
		const reasons: (string | undefined)[] = [];

		for (const finish of ['length', 'stop']) {
			await ChatService.handleStreamResponse(
				answer(finish),
				undefined,
				(_content, _reasoning, _timings, _toolCalls, finishReason) => reasons.push(finishReason)
			);
		}

		expect(reasons).toEqual(['length', 'stop']);
	});
});
