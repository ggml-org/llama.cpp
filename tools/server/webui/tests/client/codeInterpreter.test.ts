import { describe, it, expect } from 'vitest';
import '$lib/services/tools';
import { AGENTIC_TAGS } from '$lib/constants';
import { SETTING_CONFIG_DEFAULT } from '$lib/constants/settings-config';
import { findToolByName } from '$lib/services/tools/registry';
import { runCodeInterpreter } from '$lib/services/tools/codeInterpreter';
import { MessageRole, ToolCallType } from '$lib/enums';
import {
	CODE_CONTEXT_CLEAR_JS_TOOL_NAME,
	CODE_CONTEXT_DEFINE_JS_TOOL_NAME,
	CODE_INTERPRETER_JS_TOOL_NAME,
	getJavaScriptSourceArgument
} from '$lib/services/tools/codeInterpreterSharedState';
import type { AgenticMessage } from '$lib/types/agentic';

describe('code_interpreter_javascript_execute tool (browser Worker)', () => {
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

	it('returns the last expression when multiple top-level statements share one line', async () => {
		const code =
			'let primes=[];let n=2;function isPrime(x){if(x<2)return false;for(let i=2;i*i<=x;i++){if(x%i===0)return false;}return true;}while(primes.length<10){if(isPrime(n))primes.push(n);n++;}primes;';

		const { result, error } = await runCodeInterpreter(code, 3000);

		expect(error).toBeUndefined();
		expect(result).toBe(
			JSON.stringify([2, 3, 5, 7, 11, 13, 17, 19, 23, 29], null, 2)
		);
	});

	it('supports function declaration followed by a call on the same line', async () => {
		const code = 'function fact(n){return n===0?1:n*fact(n-1);} fact(10);';

		const { result, error } = await runCodeInterpreter(code, 3000);

		expect(error).toBeUndefined();
		expect(result).toBe('3628800');
	});

	it('recovers from over-escaped multiline snippets passed through tool arguments', async () => {
		const tool = findToolByName('code_interpreter_javascript_execute');
		expect(tool).toBeDefined();

		const code = String.raw`// Generate first 10 Fibonacci numbers\nfunction fib(n) {\n  const arr = [0, 1];\n  for (let i = 2; i < n; i++) {\n    arr[i] = arr[i-1] + arr[i-2];\n  }\n  return arr.slice(0, n);\n}\nfib(10);`;
		const res = await tool!.execute(JSON.stringify({ code }));

		expect(JSON.parse(res.content)).toStrictEqual([0, 1, 1, 2, 3, 5, 8, 13, 21, 34]);
	});

	it('extracts partial streamed JavaScript tool arguments before the JSON object is complete', () => {
		const partialArgs =
			'{"code":"// Generate first 3 Fibonacci numbers\\nfunction fib(n) {\\n  const arr = [0, 1];\\n  return arr.slice(0, n);\\n}\\nfib(3);';

		expect(getJavaScriptSourceArgument(CODE_INTERPRETER_JS_TOOL_NAME, partialArgs)).toBe(`// Generate first 3 Fibonacci numbers
function fib(n) {
  const arr = [0, 1];
  return arr.slice(0, n);
}
fib(3);`);
	});

	it('extracts partial streamed JavaScript tool arguments with escaped quotes intact', () => {
		const partialArgs = '{"code":"console.log(\\"hello\\");\\nconst label = \\"world';

		expect(getJavaScriptSourceArgument(CODE_INTERPRETER_JS_TOOL_NAME, partialArgs)).toBe(
			'console.log("hello");\nconst label = "world'
		);
	});

	it('uses the latest shared JavaScript state for later interpreter calls', async () => {
		const tool = findToolByName('code_interpreter_javascript_execute');
		expect(tool).toBeDefined();

		const messages: AgenticMessage[] = [
			{
				role: MessageRole.ASSISTANT,
				tool_calls: [
					{
						id: 'define-1',
						type: ToolCallType.FUNCTION,
						function: {
							name: CODE_CONTEXT_DEFINE_JS_TOOL_NAME,
							arguments: JSON.stringify({ code: 'const value = 21;' })
						}
					}
				]
			},
			{ role: MessageRole.TOOL, tool_call_id: 'define-1', content: 'Shared JavaScript state updated.' },
			{
				role: MessageRole.ASSISTANT,
				tool_calls: [
					{
						id: 'define-2',
						type: ToolCallType.FUNCTION,
						function: {
							name: CODE_CONTEXT_DEFINE_JS_TOOL_NAME,
							arguments: JSON.stringify({ code: 'const value = 34;' })
						}
					}
				]
			},
			{ role: MessageRole.TOOL, tool_call_id: 'define-2', content: 'Shared JavaScript state updated.' }
		];

		const res = await tool!.execute(
			JSON.stringify({ code: 'value;' }),
			SETTING_CONFIG_DEFAULT,
			{ messages }
		);

		expect(res.isError).toBe(false);
		expect(res.content).toBe('34');
	});

	it('reuses shared JavaScript state reconstructed from a persisted assistant transcript', async () => {
		const tool = findToolByName('code_interpreter_javascript_execute');
		expect(tool).toBeDefined();

		const persistedToolArgs = JSON.stringify({ code: 'const value = 55;' });
		const persistedAssistantContent = `${AGENTIC_TAGS.TOOL_CALL_START}
${AGENTIC_TAGS.TOOL_NAME_PREFIX}${CODE_CONTEXT_DEFINE_JS_TOOL_NAME}${AGENTIC_TAGS.TAG_SUFFIX}
${AGENTIC_TAGS.TOOL_ARGS_START}${persistedToolArgs}${AGENTIC_TAGS.TOOL_ARGS_END}
Shared JavaScript state updated.
${AGENTIC_TAGS.TOOL_CALL_END}`;

		const messages: AgenticMessage[] = [
			{
				role: MessageRole.ASSISTANT,
				content: persistedAssistantContent,
				tool_calls: [
					{
						id: 'define-1',
						type: ToolCallType.FUNCTION,
						function: {
							name: CODE_CONTEXT_DEFINE_JS_TOOL_NAME,
							arguments: persistedToolArgs
						}
					}
				]
			}
		];

		const res = await tool!.execute(
			JSON.stringify({ code: 'value;' }),
			SETTING_CONFIG_DEFAULT,
			{ messages }
		);

		expect(res.isError).toBe(false);
		expect(res.content).toBe('55');
	});

	it('stops using shared JavaScript state after a clear tool call', async () => {
		const tool = findToolByName('code_interpreter_javascript_execute');
		expect(tool).toBeDefined();

		const messages: AgenticMessage[] = [
			{
				role: MessageRole.ASSISTANT,
				tool_calls: [
					{
						id: 'define-1',
						type: ToolCallType.FUNCTION,
						function: {
							name: CODE_CONTEXT_DEFINE_JS_TOOL_NAME,
							arguments: JSON.stringify({ code: 'const value = 21;' })
						}
					}
				]
			},
			{ role: MessageRole.TOOL, tool_call_id: 'define-1', content: 'Shared JavaScript state updated.' },
			{
				role: MessageRole.ASSISTANT,
				tool_calls: [
					{
						id: 'clear-1',
						type: ToolCallType.FUNCTION,
						function: {
							name: CODE_CONTEXT_CLEAR_JS_TOOL_NAME,
							arguments: JSON.stringify({})
						}
					}
				]
			},
			{ role: MessageRole.TOOL, tool_call_id: 'clear-1', content: 'Shared JavaScript state cleared.' }
		];

		const res = await tool!.execute(
			JSON.stringify({ code: 'typeof value;' }),
			SETTING_CONFIG_DEFAULT,
			{ messages }
		);

		expect(res.isError).toBe(false);
		expect(res.content).toBe('undefined');
	});

	it('validates shared JavaScript state updates before accepting them', async () => {
		const tool = findToolByName(CODE_CONTEXT_DEFINE_JS_TOOL_NAME);
		expect(tool).toBeDefined();

		const res = await tool!.execute(
			JSON.stringify({ code: 'function broken() { return 1 + ; }' }),
			SETTING_CONFIG_DEFAULT
		);

		expect(res.isError).toBe(true);
		expect(res.content).toContain('Error');
		expect(res.content).toContain('line 1');
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
		const tool = findToolByName('code_interpreter_javascript_execute');
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
		const tool = findToolByName('code_interpreter_javascript_execute');
		expect(tool).toBeDefined();

		const cfg = { ...SETTING_CONFIG_DEFAULT, codeInterpreterTimeoutSeconds: 0.05 };
		const res = await tool!.execute(JSON.stringify({ code: '(() => { while (true) {} })()' }), cfg);

		expect(res.content).toContain('Timed out');
	});

	it('returns a more actionable no-output hint when nothing was produced', async () => {
		const tool = findToolByName('code_interpreter_javascript_execute');
		expect(tool).toBeDefined();

		const res = await tool!.execute(JSON.stringify({ code: 'if (true) { const x = 1; }' }));

		expect(res.content).toContain('final expression');
		expect(res.content).toContain('console.log');
		expect(res.content).toContain('await it at top level');
	});
});
