/**
 * A canned llama-server for demos and e2e checks that should not need a real one, or a model:
 * routes its endpoints in the browser, so nothing reaches a server and no model loads.
 */

import type { BrowserContext, Page } from '@playwright/test';

export interface MockServerOptions {
	/** Model ids the router lists, all loaded. */
	models?: string[];
	/** The assistant's reply to every message, streamed a word per chunk. */
	reply?: string;
}

function completionStream(reply: string): string {
	const words = reply.match(/\S+\s*/g) ?? [];
	const chunk = (delta: object, finish: null | string = null) =>
		`data: ${JSON.stringify({ choices: [{ delta, finish_reason: finish, index: 0 }], object: 'chat.completion.chunk' })}\n\n`;

	return [
		chunk({ role: 'assistant' }),
		...words.map((word) => chunk({ content: word })),
		chunk({}, 'stop'),
		'data: [DONE]\n\n'
	].join('');
}

/** Call before the page opens; a demo sets `test.use({ mockServer: {...} })` instead. */
export async function mockLlamaServer(
	target: BrowserContext | Page,
	{
		models = ['ggml-org/gemma-3-4b-it-GGUF:Q4_K_M'],
		reply = 'Hello! How can I help?'
	}: MockServerOptions = {}
): Promise<void> {
	const data = models.map((id) => ({
		id,
		object: 'model',
		owned_by: 'llamacpp',
		status: { value: 'loaded' }
	}));

	await target.route(/\/props(\?|$)/, (route) =>
		route.fulfill({
			json: { build_info: 'mock', model_path: 'none', role: 'router', ui_settings: {} }
		})
	);
	await target.route(/\/(v1\/)?models(\?|$)/, (route) =>
		route.fulfill({ json: { data, object: 'list' } })
	);
	await target.route(/\/models\/(load|unload)$/, (route) =>
		route.fulfill({ json: { success: true } })
	);
	await target.route(/\/models\/sse$/, (route) => route.abort());
	await target.route(/\/tools(\?|$)/, (route) => route.fulfill({ json: [] }));
	await target.route(/\/v1\/streams\/lookup$/, (route) => route.fulfill({ status: 404 }));
	await target.route(/\/v1\/chat\/completions$/, (route) =>
		route.fulfill({
			body: completionStream(reply),
			headers: { 'content-type': 'text/event-stream' }
		})
	);
}
