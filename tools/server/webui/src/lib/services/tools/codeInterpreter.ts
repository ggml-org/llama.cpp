import type { ApiToolDefinition } from '$lib/types';
import type { ToolSettingDefinition } from '$lib/services/tools/registry';
import { registerTool } from './registry';
import { Parser } from 'acorn';
import type { Options as AcornOptions } from 'acorn';

export const CODE_INTERPRETER_JS_TOOL_NAME = 'code_interpreter_javascript';

export const codeInterpreterToolDefinition: ApiToolDefinition = {
	type: 'function',
	function: {
		name: CODE_INTERPRETER_JS_TOOL_NAME,
		description:
			'Execute JavaScript in a sandboxed Worker. Your code runs inside an async function (top-level await is supported). Do not wrap code in an async IIFE like (async () => { ... })() unless you return/await it, otherwise the tool may finish before async logs run. If you use promises, they must be awaited. Returns combined console output and the final evaluated value. (no output) likely indicates either an unawaited promise or that you did not output anything.',
		parameters: {
			type: 'object',
			properties: {
				code: {
					type: 'string',
					description: 'JavaScript source code to run.'
				}
			},
			required: ['code']
		}
	}
};

export interface CodeInterpreterResult {
	result?: string;
	logs: string[];
	error?: string;
	errorLine?: number;
	errorLineContent?: string;
	errorStack?: string;
	errorFrame?: string;
	errorColumn?: number;
}

export async function runCodeInterpreter(
	code: string,
	timeoutMs = 30_000
): Promise<CodeInterpreterResult> {
	// Pre-parse in the main thread so syntax errors include line/column and source line.
	// (V8 SyntaxError stacks from eval/new Function are often missing user-code locations.)
	// Lines before user code in wrapped string:
	// 1: (async () => {
	// 2: "use strict";
	// 3: // __USER_CODE_START__
	const USER_OFFSET = 3;
	try {
		const sourceName = CODE_INTERPRETER_JS_TOOL_NAME;
		const wrappedForParse =
			`(async () => {\n` +
			`"use strict";\n` +
			`// __USER_CODE_START__\n` +
			`${code ?? ''}\n` +
			`// __USER_CODE_END__\n` +
			`})()\n` +
			`//# sourceURL=${sourceName}\n`;

		const acornOptions: AcornOptions = {
			ecmaVersion: 2024,
			sourceType: 'script',
			allowAwaitOutsideFunction: true,
			locations: true
		};
		Parser.parse(wrappedForParse, acornOptions);
	} catch (err) {
		const message = err instanceof Error ? err.message : String(err);
		const loc = (err as Error & { loc?: { line: number; column: number } }).loc;
		const userLine = loc?.line ? Math.max(1, loc.line - USER_OFFSET) : undefined;
		const userColumn = typeof loc?.column === 'number' ? loc.column + 1 : undefined;
		const lines = (code ?? '').split('\n');
		const lineContent = userLine ? lines[userLine - 1]?.trim() : undefined;
		return {
			logs: [],
			error: message,
			errorLine: userLine,
			errorColumn: userColumn,
			errorLineContent: lineContent
		};
	}

	return new Promise((resolve) => {
		const logs: string[] = [];

		const workerSource = `
      const send = (msg) => postMessage(msg);

      const logs = [];
      ['log','info','warn','error'].forEach((level) => {
        const orig = console[level];
        console[level] = (...args) => {
          const text = args.map((a) => {
            try { return typeof a === 'string' ? a : JSON.stringify(a); }
            catch { return String(a); }
          }).join(' ');
          logs.push(text);
          send({ type: 'log', level, text });
          if (orig) try { orig.apply(console, args); } catch { /* ignore */ }
        };
      });

      const transformCode = (code) => {
        const lines = (code ?? '').split('\\n');
        let i = lines.length - 1;
        while (i >= 0 && lines[i].trim() === '') i--;
        if (i < 0) return code ?? '';

        const last = lines[i];
        const trimmed = last.trim();

        // If already returns, leave as-is.
        if (/^return\\b/.test(trimmed)) return code ?? '';

        // If the last line starts/ends with block delimiters, keep code as-is (likely a statement block).
        if (/^[}\\])]/.test(trimmed) || trimmed.endsWith('{') || trimmed.endsWith('};')) {
          return code ?? '';
        }

        // If it's a declaration, return that identifier.
        const declMatch = trimmed.match(/^(const|let|var)\\s+([A-Za-z_$][\\w$]*)/);
        if (declMatch) {
          const name = declMatch[2];
          lines.push(\`return \${name};\`);
          return lines.join('\\n');
        }

        // Default: treat last statement as expression and return it.
        lines[i] = \`return (\${trimmed.replace(/;$/, '')});\`;
        return lines.join('\\n');
      };

      const run = async (code) => {
        try {
          const executable = transformCode(code);
          const markerStart = '__USER_CODE_START__';
          const markerEnd = '__USER_CODE_END__';
          // Use eval within an async IIFE so we can use return and get better syntax error locations.
          // Lines before user code in wrapped string:
          // 1: (async () => {
          // 2: "use strict";
          // 3: // __USER_CODE_START__
          const USER_OFFSET = 3;
          const sourceName = 'code_interpreter_javascript';
          const wrapped =
            \`(async () => {\\n\` +
            \`"use strict";\\n\` +
            \`// \${markerStart}\\n\` +
            \`\${executable}\\n\` +
            \`// \${markerEnd}\\n\` +
            \`})()\\n\` +
            \`//# sourceURL=\${sourceName}\\n\`;
          // eslint-disable-next-line no-eval
          const result = await eval(wrapped);
          send({ type: 'done', result, logs });
        } catch (err) {
          let lineNum = undefined;
          let lineText = undefined;
          let columnNum = undefined;
          try {
            const stack = String(err?.stack ?? '');
            const match =
              stack.match(/(?:<anonymous>|code_interpreter_javascript):(\\d+):(\\d+)/) ||
              stack.match(/:(\\d+):(\\d+)/); // fallback: first frame with line/col
            if (match) {
              const rawLine = Number(match[1]);
              const rawCol = Number(match[2]);
              // Our wrapped string puts user code starting at line USER_OFFSET + 1
              const userLine = Math.max(1, rawLine - USER_OFFSET);
              lineNum = userLine;
              columnNum = rawCol;
              const srcLines = (code ?? '').split('\\n');
              lineText = srcLines[userLine - 1]?.trim();
            }
          } catch {}
          if (!lineNum && err?.message) {
            const idMatch = String(err.message).match(/['"]?([A-Za-z_$][\\w$]*)['"]? is not defined/);
            if (idMatch) {
              const ident = idMatch[1];
              const srcLines = (code ?? '').split('\\n');
              const foundIdx = srcLines.findIndex((l) => l.includes(ident));
              if (foundIdx !== -1) {
                lineNum = foundIdx + 1;
                lineText = srcLines[foundIdx]?.trim();
              }
            }
          }
          if (!lineNum) {
            const ln = err?.lineNumber ?? err?.lineno ?? err?.line;
            if (typeof ln === 'number') {
              lineNum = ln;
              const srcLines = (code ?? '').split('\\n');
              lineText = srcLines[ln - 1]?.trim();
            }
          }
          if (columnNum === undefined) {
            const col = err?.columnNumber ?? err?.colno ?? undefined;
            if (typeof col === 'number') columnNum = col;
          }
          const stack = err?.stack ? String(err.stack) : undefined;
          const firstStackFrame = stack
            ?.split('\\n')
            .find((l) => l.includes('<anonymous>') || l.includes('code_interpreter_javascript'));
          send({
            type: 'error',
            message: err?.message ?? String(err),
            stack,
            frame: firstStackFrame,
            logs,
            line: lineNum,
            lineContent: lineText,
            column: columnNum
          });
        }
      };

      self.onmessage = (e) => {
        run(e.data?.code ?? '');
      };
    `;

		const blob = new Blob([workerSource], { type: 'application/javascript' });
		const worker = new Worker(URL.createObjectURL(blob));

		const timer = setTimeout(() => {
			worker.terminate();
			resolve({ logs, error: 'Timed out' });
		}, timeoutMs);

		worker.onmessage = (event: MessageEvent) => {
			const { type } = event.data || {};
			if (type === 'log') {
				logs.push(event.data.text);
				return;
			}

			clearTimeout(timer);
			worker.terminate();

			if (type === 'error') {
				resolve({
					logs: event.data.logs ?? logs,
					error: event.data.message,
					errorLine: event.data.line,
					errorLineContent: event.data.lineContent,
					errorStack: event.data.stack,
					errorFrame: event.data.frame,
					errorColumn: event.data.column
				});
				return;
			}

			if (type === 'done') {
				const value = event.data.result;
				let rendered = '';
				try {
					rendered = typeof value === 'string' ? value : JSON.stringify(value, null, 2);
				} catch {
					rendered = String(value);
				}
				resolve({ logs: event.data.logs ?? logs, result: rendered });
			}
		};

		worker.postMessage({ code });
	});
}

registerTool({
	name: CODE_INTERPRETER_JS_TOOL_NAME,
	label: 'Code Interpreter (JavaScript)',
	description: 'Run JavaScript in a sandboxed Worker and capture logs plus final value.',
	enableConfigKey: 'enableCodeInterpreterTool',
	defaultEnabled: true,
	settings: [
		{
			key: 'codeInterpreterTimeoutSeconds',
			label: 'Code interpreter timeout (seconds)',
			type: 'input',
			defaultValue: 30,
			help: 'Maximum time allowed for the JavaScript tool to run before it is terminated.'
		} satisfies ToolSettingDefinition
	],
	definition: codeInterpreterToolDefinition,
	execute: async (argsJson: string, config) => {
		let code = argsJson;
		try {
			const parsedArgs = JSON.parse(argsJson);
			if (parsedArgs && typeof parsedArgs === 'object' && typeof parsedArgs.code === 'string') {
				code = parsedArgs.code;
			}
		} catch {
			// leave raw
		}

		const timeoutSecondsRaw = config?.codeInterpreterTimeoutSeconds;
		const timeoutSeconds =
			typeof timeoutSecondsRaw === 'number'
				? timeoutSecondsRaw
				: typeof timeoutSecondsRaw === 'string'
					? Number(timeoutSecondsRaw)
					: 30;
		const timeoutMs = Number.isFinite(timeoutSeconds)
			? Math.max(0, Math.round(timeoutSeconds * 1000))
			: 30_000;

		const {
			result,
			logs,
			error,
			errorLine,
			errorLineContent,
			errorStack,
			errorFrame,
			errorColumn
		} = await runCodeInterpreter(code, timeoutMs);
		let combined = '';
		if (logs?.length) combined += logs.join('\n');
		if (combined && (result !== undefined || error)) combined += '\n';
		if (error) {
			const lineLabel = errorLine !== undefined ? `line ${errorLine}` : null;
			const columnLabel =
				errorLine !== undefined && typeof errorColumn === 'number' ? `, col ${errorColumn}` : '';
			const lineSnippet =
				errorLine !== undefined && errorLineContent ? `: ${errorLineContent.trim()}` : '';
			const lineInfo = lineLabel ? ` (${lineLabel}${columnLabel}${lineSnippet})` : '';
			combined += `Error${lineInfo}: ${error}`;
			if (!lineLabel) {
				if (errorFrame) {
					combined += `\nFrame: ${errorFrame}`;
				} else if (errorStack) {
					combined += `\nStack: ${errorStack}`;
				}
			}
		} else if (result !== undefined) {
			combined += result;
		} else if (!combined) {
			combined = '(no output, did you forget to await a top level promise?)';
		}
		return { content: combined };
	}
});
