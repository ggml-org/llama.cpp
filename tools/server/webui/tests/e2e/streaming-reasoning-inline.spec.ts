import { test, expect } from '@playwright/test';

/**
 * End-to-end regression that reproduces the real streaming bug reported by users:
 * - The model streams reasoning → tool call → (new request) reasoning → final answer.
 * - We only mock the HTTP API; the UI, stores, and client-side tool execution run unchanged.
 * - The test asserts that the second reasoning chunk becomes visible *while the second
 *   completion stream is still open* (i.e., without a page refresh and before final content).
 */
test('reasoning -> tool -> reasoning streams inline without refresh', async ({ page }) => {
	// Install fetch stub & config before the app loads
	await page.addInitScript(() => {
		// Enable the calculator tool client-side
		localStorage.setItem(
			'LlamaCppWebui.config',
			JSON.stringify({ enableCalculatorTool: true, showToolCalls: true })
		);

		let completionCall = 0;
		let secondController: ReadableStreamDefaultController | null = null;
		const encoder = new TextEncoder();

		const originalFetch = window.fetch.bind(window);
		const w = window as unknown as {
			__completionCallCount?: number;
			__flushSecondStream?: () => void;
		};
		w.__completionCallCount = 0;

		window.fetch = (input: RequestInfo | URL, init?: RequestInit) => {
			const url = input instanceof Request ? input.url : String(input);

			// Mock minimal server props & model list
			if (url.includes('/props')) {
				return Promise.resolve(
					new Response(
						JSON.stringify({
							role: 'model',
							system_prompt: null,
							default_generation_settings: { params: {}, n_ctx: 4096 }
						}),
						{ headers: { 'Content-Type': 'application/json' }, status: 200 }
					)
				);
			}

			if (url.includes('/v1/models')) {
				return Promise.resolve(
					new Response(
						JSON.stringify({ object: 'list', data: [{ id: 'mock-model', object: 'model' }] }),
						{ headers: { 'Content-Type': 'application/json' }, status: 200 }
					)
				);
			}

			// Mock the streaming chat completions endpoint
			if (url.includes('/v1/chat/completions')) {
				completionCall += 1;
				w.__completionCallCount = completionCall;

				// First request: reasoning + tool call, then DONE
				if (completionCall === 1) {
					const stream = new ReadableStream({
						start(controller) {
							controller.enqueue(
								encoder.encode(
									`data: ${JSON.stringify({
										choices: [{ delta: { reasoning_content: 'reasoning-step-1' } }]
									})}\n\n`
								)
							);
							controller.enqueue(
								encoder.encode(
									`data: ${JSON.stringify({
										choices: [
											{
												delta: {
													tool_calls: [
														{
															id: 'call-1',
															type: 'function',
															function: { name: 'calculator', arguments: '{"expression":"1+1"}' }
														}
													]
												}
											}
										]
									})}\n\n`
								)
							);
							controller.enqueue(encoder.encode('data: [DONE]\n\n'));
							controller.close();
						}
					});

					return Promise.resolve(
						new Response(stream, {
							headers: { 'Content-Type': 'text/event-stream' },
							status: 200
						})
					);
				}

				// Second request: stream reasoning, leave stream open until test flushes final content
				if (completionCall === 2) {
					const stream = new ReadableStream({
						start(controller) {
							secondController = controller;
							controller.enqueue(
								encoder.encode(
									`data: ${JSON.stringify({
										choices: [{ delta: { reasoning_content: 'reasoning-step-2' } }]
									})}\n\n`
								)
							);
							// DO NOT close yet – test will push final content later.
						}
					});

					// expose a helper so the test can finish the stream after the assertion
					w.__flushSecondStream = () => {
						if (!secondController) return;
						secondController.enqueue(
							encoder.encode(
								`data: ${JSON.stringify({
									choices: [{ delta: { content: 'final-answer' } }]
								})}\n\n`
							)
						);
						secondController.enqueue(encoder.encode('data: [DONE]\n\n'));
						secondController.close();
					};

					return Promise.resolve(
						new Response(stream, {
							headers: { 'Content-Type': 'text/event-stream' },
							status: 200
						})
					);
				}
			}

			// Fallback to real fetch for everything else
			return originalFetch(input, init);
		};
	});

	// Launch the UI
	await page.goto('http://localhost:8181/');

	// Send a user message to trigger streaming
	const textarea = page.getByPlaceholder('Ask anything...');
	await textarea.fill('test message');
	await page.getByRole('button', { name: 'Send' }).click();

	// Expand the reasoning block so hidden text becomes visible
	const reasoningToggle = page.getByRole('button', { name: /Reasoning/ });
	await expect(reasoningToggle).toBeVisible({ timeout: 5000 });
	await reasoningToggle.click();

	// Wait for first reasoning chunk to appear (UI)
	await expect
		.poll(async () =>
			page.locator('[aria-label="Assistant message with actions"]').first().innerText()
		)
		.toContain('reasoning-step-1');

	// Wait for tool result (calculator executed client-side)
	await expect(page.getByText('2', { exact: true })).toBeVisible({ timeout: 5000 });

	// Ensure the follow-up completion request (after tool execution) was actually triggered
	await expect
		.poll(() =>
			page.evaluate(
				() => (window as unknown as { __completionCallCount?: number }).__completionCallCount || 0
			)
		)
		.toBeGreaterThanOrEqual(2);

	// Critical assertion: the second reasoning chunk should appear while the second stream is still open
	await expect
		.poll(async () =>
			page.locator('[aria-label="Assistant message with actions"]').first().innerText()
		)
		.toContain('reasoning-step-2');

	// Finish streaming the final content and verify it appears
	await page.evaluate(() =>
		(window as unknown as { __flushSecondStream?: () => void }).__flushSecondStream?.()
	);
	await expect(page.getByText('final-answer').first()).toBeVisible({ timeout: 5000 });
});
