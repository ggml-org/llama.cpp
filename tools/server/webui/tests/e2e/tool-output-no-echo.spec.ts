import { test, expect } from '@playwright/test';

const streamBody = (...lines: string[]) => lines.map((l) => `data: ${l}\n\n`).join('');

test('tool output does not echo tool arguments back to the model', async ({ page }) => {
	const LARGE_CODE = [
		'// LARGE_CODE_BEGIN',
		'function findMaxL(outerWidth, outerHeight, outerDepth, innerWidth, innerHeight) {',
		'  const outerDiagonal = Math.sqrt(outerWidth**2 + outerHeight**2 + outerDepth**2);',
		'  const maxL = Math.sqrt(outerDiagonal**2 - innerWidth**2 - innerHeight**2);',
		'  return maxL;',
		'}',
		'findMaxL(98, 76, 52, 6, 6);',
		'// LARGE_CODE_END',
		'',
		'// filler to simulate large prompt without breaking last-expression transform',
		`const __filler = "${'x'.repeat(4000)}";`,
		'1 + 1'
	].join('\n');

	// Ensure tool is enabled before app boot
	await page.addInitScript(
		(cfg) => {
			localStorage.setItem('LlamaCppWebui.config', JSON.stringify(cfg));
		},
		{
			enableCodeInterpreterTool: true,
			showToolCalls: true
		}
	);

	// Mock /props and model list so UI initializes in static preview
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
	await page.route('**/v1/models**', async (route) => {
		await route.fulfill({
			status: 200,
			contentType: 'application/json',
			body: JSON.stringify({ object: 'list', data: [{ id: 'mock-model', object: 'model' }] })
		});
	});

	type ChatCompletionRequestBody = { messages?: unknown };
	let secondRequestBody: ChatCompletionRequestBody | string | null = null;
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
									function: {
										name: 'code_interpreter_javascript',
										arguments: JSON.stringify({ code: LARGE_CODE })
									}
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

		// Second completion request should include tool output. Capture request body for assertions.
		try {
			secondRequestBody = route.request().postDataJSON() as ChatCompletionRequestBody;
		} catch {
			secondRequestBody = route.request().postData();
		}

		const chunk3 = JSON.stringify({
			choices: [{ delta: { content: 'final-answer' } }]
		});
		const body = streamBody(chunk3, '[DONE]');
		completionCall++;
		await route.fulfill({
			status: 200,
			contentType: 'text/event-stream',
			body
		});
	});

	await page.goto('http://localhost:8181/');

	// Send a user message to trigger streaming
	const textarea = page.getByPlaceholder('Ask anything...');
	await expect(textarea).toBeVisible();
	await textarea.fill('run js');
	await page.getByRole('button', { name: 'Send' }).click();

	await expect(page.getByText('final-answer')).toBeVisible({ timeout: 10000 });
	expect(completionCall).toBe(2);
	expect(secondRequestBody).toBeTruthy();
	if (!secondRequestBody || typeof secondRequestBody === 'string') {
		throw new Error('Expected second completion request body JSON');
	}

	// Assert the second request contains tool output but does NOT duplicate the large code in the tool output content.
	const secondJson = secondRequestBody as unknown as ChatCompletionRequestBody;
	const messages = secondJson.messages;
	expect(Array.isArray(messages)).toBe(true);
	const typedMessages = messages as Array<Record<string, unknown>>;

	const toolMessage = typedMessages.find((m) => m.role === 'tool' && m.tool_call_id === 'call-1');
	expect(toolMessage).toBeTruthy();
	expect(String(toolMessage?.content ?? '')).not.toContain('LARGE_CODE_BEGIN');
	expect(String(toolMessage?.content ?? '')).not.toContain('function findMaxL');

	// The original tool call arguments are still present in the assistant tool call message.
	const assistantWithToolCall = typedMessages.find(
		(m) => m.role === 'assistant' && Array.isArray(m.tool_calls)
	);
	expect(assistantWithToolCall).toBeTruthy();
	expect(JSON.stringify(assistantWithToolCall?.tool_calls ?? null)).toContain('LARGE_CODE_BEGIN');
	// Preserve the model's reasoning across tool-call resumptions (required for gpt-oss).
	expect(String(assistantWithToolCall?.reasoning_content ?? '')).toContain('reasoning-step-1');
});
