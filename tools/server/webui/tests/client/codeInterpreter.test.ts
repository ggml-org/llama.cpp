import { describe, it, expect } from 'vitest';
import '$lib/services/tools';
import { SETTING_CONFIG_DEFAULT } from '$lib/constants/settings-config';
import { findToolByName } from '$lib/services/tools/registry';
import { runCodeInterpreter } from '$lib/services/tools/codeInterpreter';

describe('code_interpreter_javascript tool (browser Worker)', () => {
	it('evaluates JS and returns final value', async () => {
		const code = `
      const f1 = 2.8;
      const f2 = 4.5;
      const ratio = (f2 / f1) ** 2;
      const stops = Math.log2(ratio);
      stops;
    `;

		const { result, error, logs } = await runCodeInterpreter(code, 3000);

		expect(error).toBeUndefined();
		expect(logs).toStrictEqual([]);
		expect(result).toBeDefined();
		expect(Number(result)).toBeCloseTo(1.3689963, 5);
	});

	it('captures console output', async () => {
		const code = `
      console.log('hello', 1+1);
      return 42;
    `;

		const { result, error, logs } = await runCodeInterpreter(code, 3000);

		expect(error).toBeUndefined();
		expect(logs).toContain('hello 2');
		expect(result).toBe('42');
	});

	it('handles block-ending scripts (FizzBuzz loop) without forcing a return', async () => {
		const code = `
      for (let i = 1; i <= 20; i++) {
        let out = '';
        if (i % 3 === 0) out += 'Fizz';
        if (i % 5 === 0) out += 'Buzz';
        console.log(out || i);
      }
    `;

		const { error, logs, result } = await runCodeInterpreter(code, 3000);

		expect(error).toBeUndefined();
		// Should have produced logs, but no forced result
		expect(logs.length).toBeGreaterThan(0);
		expect(logs[0]).toBe('1');
		expect(result === undefined || result === '').toBe(true);
	});

	it('reports line number and content on error', async () => {
		const code = `
      const x = 1;
      const y = oops; // ReferenceError
      return x + y;
    `;

		const { error, errorLine, errorLineContent, errorStack, errorFrame } = await runCodeInterpreter(
			code,
			3000
		);

		expect(error).toBeDefined();
		expect(errorLine).toBeGreaterThan(0);
		expect(errorLineContent ?? '').toBeTypeOf('string');
		expect(errorFrame || errorStack || '').toSatisfy((s: string) => s.length > 0);
	});

	it('includes at least a frame for syntax errors without line capture', async () => {
		const code = `
      function broken() {
        const a = 1;
        const b = 2;
        return a + ; // syntax error
      }
      broken();
    `;

		const { error, errorLine, errorLineContent, errorFrame, errorStack } = await runCodeInterpreter(
			code,
			3000
		);

		expect(error).toBeDefined();
		expect(
			errorLine !== undefined ||
				(errorLineContent && errorLineContent.length > 0) ||
				(errorFrame && errorFrame.length > 0) ||
				(errorStack && errorStack.length > 0)
		).toBe(true);
	});

	it('captures line number and source line for a missing parenthesis syntax error', async () => {
		const code = [
			'function f() {',
			'  const x = Math.max(1, 2; // missing )',
			'  return x;',
			'}',
			'f();'
		].join('\n');

		const { error, errorLine, errorLineContent } = await runCodeInterpreter(code, 3000);

		expect(error).toBeDefined();
		expect(errorLine).toBe(2);
		expect(errorLineContent || '').toContain('Math.max');
	});

	it('includes line and snippet in the tool output string for syntax errors', async () => {
		const tool = findToolByName('code_interpreter_javascript');
		expect(tool).toBeDefined();

		const code = [
			'function f() {',
			'  const x = Math.max(1, 2; // missing )',
			'  return x;',
			'}',
			'f();'
		].join('\n');

		const res = await tool!.execute(JSON.stringify({ code }));
		expect(res.content).toContain('Error');
		expect(res.content).toContain('line 2');
		expect(res.content).toContain('Math.max');
	});

	it('respects the configured timeout setting', async () => {
		const tool = findToolByName('code_interpreter_javascript');
		expect(tool).toBeDefined();

		const cfg = { ...SETTING_CONFIG_DEFAULT, codeInterpreterTimeoutSeconds: 0.05 };
		const res = await tool!.execute(JSON.stringify({ code: '(() => { while (true) {} })()' }), cfg);

		expect(res.content).toContain('Timed out');
	});
});
