import { test, expect } from '@playwright/test';

// Helper to build a streaming body with newline-delimited "data:" lines
const streamBody = (...lines: string[]) => lines.map((l) => `data: ${l}\n\n`).join('');

test('reasoning + tool + final content stream into one reasoning block', async ({ page }) => {
	page.on('console', (msg) => {
		console.log('BROWSER LOG:', msg.type(), msg.text());
	});
	page.on('pageerror', (err) => {
		console.log('BROWSER ERROR:', err.message);
	});

	// Inject fetch stubs early so server props/models succeed before the app initializes
	await page.addInitScript(() => {
		// Ensure calculator tool is enabled so tool calls are processed client-side
		localStorage.setItem(
			'LlamaCppWebui.config',
			JSON.stringify({ enableCalculatorTool: true, showToolCalls: true })
		);

		const propsBody = {
			role: 'model',
			system_prompt: null,
			default_generation_settings: { params: {}, n_ctx: 4096 }
		};
		const modelsBody = { object: 'list', data: [{ id: 'mock-model', object: 'model' }] };
		const originalFetch = window.fetch;
		window.fetch = (...args) => {
			const url = args[0] instanceof Request ? args[0].url : String(args[0]);
			if (url.includes('/props')) {
				return Promise.resolve(
					new Response(JSON.stringify(propsBody), {
						status: 200,
						headers: { 'Content-Type': 'application/json' }
					})
				);
			}
			if (url.includes('/v1/models')) {
				return Promise.resolve(
					new Response(JSON.stringify(modelsBody), {
						status: 200,
						headers: { 'Content-Type': 'application/json' }
					})
				);
			}
			return originalFetch(...args);
		};
	});

	// Mock /props to keep the UI enabled
	await page.route('**/props**', async (route) => {
		await route.fulfill({
			status: 200,
			contentType: 'application/json',
			body: JSON.stringify({
				role: 'model',
				system_prompt: null,
				default_generation_settings: { params: {}, n_ctx: 4096 }
			})
		});
	});

	// Mock model list
	await page.route('**/v1/models**', async (route) => {
		await route.fulfill({
			status: 200,
			contentType: 'application/json',
			body: JSON.stringify({ object: 'list', data: [{ id: 'mock-model', object: 'model' }] })
		});
	});

	// Mock the chat completions endpoint twice: first call returns reasoning+tool call,
	// second call (after tool execution) returns more reasoning + final answer.
	let completionCall = 0;
	await page.route('**/v1/chat/completions**', async (route) => {
		if (completionCall === 0) {
			const chunk1 = JSON.stringify({
				choices: [{ delta: { reasoning_content: 'reasoning-step-1' } }]
			});
			const chunk2 = JSON.stringify({
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
			});
			const body = streamBody(chunk1, chunk2, '[DONE]');
			completionCall++;
			await route.fulfill({
				status: 200,
				contentType: 'text/event-stream',
				body
			});
			return;
		}

		// Second completion: continued reasoning and final answer
		const chunk3 = JSON.stringify({
			choices: [{ delta: { reasoning_content: 'reasoning-step-2' } }]
		});
		const chunk4 = JSON.stringify({
			choices: [{ delta: { content: 'final-answer' } }]
		});
		const body = streamBody(chunk3, chunk4, '[DONE]');
		completionCall++;
		await route.fulfill({
			status: 200,
			contentType: 'text/event-stream',
			body
		});
	});

	await page.goto('http://localhost:8181/');

	// Wait for the form to be ready
	const textarea = page.getByPlaceholder('Ask anything...');
	await expect(textarea).toBeVisible();
	const sendButton = page.getByRole('button', { name: 'Send' });

	// Force-enable the input in case the UI guarded on server props during static preview
	await page.evaluate(() => {
		const ta = document.querySelector<HTMLTextAreaElement>(
			'textarea[placeholder="Ask anything..."]'
		);
		if (ta) ta.disabled = false;
		const submit = document.querySelector<HTMLButtonElement>('button[type="submit"]');
		if (submit) submit.disabled = false;
	});

	// Send a user message
	const requestPromise = page.waitForRequest('**/v1/chat/completions');
	await textarea.fill('test');

	// After typing, the send button should become enabled
	await expect(sendButton).toBeEnabled({ timeout: 5000 });

	// Click the Send button (has sr-only text "Send")
	await sendButton.click();

	await requestPromise;

	// Wait for final content to appear (streamed)
	await expect(page.getByText('final-answer')).toBeVisible({ timeout: 10000 });

	// Expand the reasoning block to make streamed reasoning visible
	const reasoningToggles = page.getByRole('button', { name: 'Reasoning' });
	const toggleCount = await reasoningToggles.count();
	for (let i = 0; i < toggleCount; i++) {
		await reasoningToggles.nth(i).click();
	}

	// Ensure both reasoning steps are present in a single reasoning block
	const reasoningBlock = page.locator('[aria-label="Assistant message with actions"]').first();
	await expect(reasoningBlock).toContainText('reasoning-step-1', { timeout: 5000 });
	await expect(reasoningBlock).toContainText('reasoning-step-2', { timeout: 5000 });

	// Tool result should be displayed (calculator result "2")
	await expect(page.getByText('2', { exact: true })).toBeVisible({ timeout: 5000 });

	expect(completionCall).toBe(2);
});
