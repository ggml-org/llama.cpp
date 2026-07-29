import { beforeEach, describe, expect, it } from 'vitest';
import { SandboxService } from '$lib/services/sandbox.service';
import { SANDBOX_TOOL_NAME } from '$lib/constants';

const run = (code: string, timeoutMs?: number) =>
	SandboxService.executeTool(SANDBOX_TOOL_NAME, {
		code,
		...(timeoutMs !== undefined ? { timeout_ms: timeoutMs } : {})
	});

describe('sandbox service', () => {
	beforeEach(async () => {
		const { settingsStore } = await import('$lib/stores/settings.svelte');
		settingsStore.config = {
			...settingsStore.config,
			symbolicMathEnabled: true
		};
	});

	it('executes plain JavaScript', async () => {
		const reply = await run('return 1 + 1;');
		expect(reply.isError).toBe(false);
		expect(reply.content).toContain('=> 2');
	});

	it('exposes nerdamer for symbolic computation', async () => {
		const reply = await run("return nerdamer.diff('sin(x)/x', 'x').toString();");
		expect(reply.isError).toBe(false);
		expect(reply.content).toContain('cos(x)');
	});

	it('computes exact rational arithmetic', async () => {
		const reply = await run("return nerdamer('1/3 + 1/6').toString();");
		expect(reply.isError).toBe(false);
		expect(reply.content).toContain('=> 1/2');
	});

	it('proves a polynomial identity symbolically', async () => {
		const reply = await run(
			"return nerdamer('expand((1+x*y)^3 - (1 + 3*x*y + 3*x^2*y^2 + x^3*y^3))').toString();"
		);
		expect(reply.isError).toBe(false);
		expect(reply.content).toContain('=> 0');
	});

	it('blocks network egress in the worker via CSP', async () => {
		const reply = await run(
			"try { await fetch('https://example.com/'); return 'leaked'; } catch { return 'blocked'; }"
		);
		expect(reply.isError).toBe(false);
		expect(reply.content).toContain('=> blocked');
	});

	it('enforces the timeout on runaway code', async () => {
		const reply = await run('while (true) {}', 500);
		expect(reply.isError).toBe(true);
		expect(reply.content).toContain('timed out');
	});
});
