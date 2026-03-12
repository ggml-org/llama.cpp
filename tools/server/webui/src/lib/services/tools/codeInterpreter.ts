import type { ApiToolDefinition } from '$lib/types';
import type {
	BuiltInToolExecutionContext,
	ToolSettingDefinition
} from '$lib/services/tools/registry';
import { SettingsFieldType } from '$lib/enums/settings';
import { registerTool } from './registry';
import { Parser } from 'acorn';
import type { Options as AcornOptions } from 'acorn';
import { maybeDecodeEscapedMultilineSource } from './codeInterpreterFormatting';
import {
	buildSharedJavaScriptContext,
	CODE_CONTEXT_CLEAR_JS_TOOL_NAME,
	CODE_CONTEXT_DEFINE_JS_TOOL_NAME,
	CODE_INTERPRETER_JS_TOOL_NAME,
	getJavaScriptSourceArgument
} from './codeInterpreterSharedState';

export const codeInterpreterToolDefinition: ApiToolDefinition = {
	type: 'function',
	function: {
		name: CODE_INTERPRETER_JS_TOOL_NAME,
		description:
			'Execute JavaScript in a sandboxed Worker. Any shared JavaScript currently stored with `code_interpreter_javascript_set_context` is inserted before this snippet automatically; use `code_interpreter_javascript_clear_context` to reset it. Ordinary declarations from the shared JavaScript are directly available here, so you usually do not need `globalThis`. If you find yourself needing the same helpers or constants again in later code interpreter calls, store them with `code_interpreter_javascript_set_context` instead of repeating them. Write ordinary top-level JavaScript: multi-line or single-line snippets are both fine, and the final top-level expression is returned automatically. Explicit return also works. Function declarations followed by calls are supported. Top-level await is supported, but promises must be awaited. Pass JavaScript source in the `code` field. Returns combined console output plus the final evaluated value. If you get "(no output)", add a final expression, explicit return, or console.log.',
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

export const codeContextDefineToolDefinition: ApiToolDefinition = {
	type: 'function',
	function: {
		name: CODE_CONTEXT_DEFINE_JS_TOOL_NAME,
		description:
			'Overwrite any existing shared JavaScript state with reusable helper code such as constants, functions, or classes. Future `code_interpreter_javascript_execute` calls automatically insert this JavaScript before their own snippet runs, so ordinary declarations here are directly available there without needing `globalThis`. Use this when you expect to need the same helpers or constants again in later code interpreter calls. Pass JavaScript source in the `code` field.',
		parameters: {
			type: 'object',
			properties: {
				code: {
					type: 'string',
					description:
						'JavaScript source code to store as the shared JavaScript state.'
				}
			},
			required: ['code']
		}
	}
};

export const codeContextClearToolDefinition: ApiToolDefinition = {
	type: 'function',
	function: {
		name: CODE_CONTEXT_CLEAR_JS_TOOL_NAME,
		description:
			'Clear the current shared JavaScript state so future `code_interpreter_javascript_execute` calls start without any previously defined helpers.',
		parameters: {
			type: 'object',
			properties: {}
		}
	}
};

export interface CodeInterpreterResult {
	result?: string;
	logs: string[];
	error?: string;
	errorScope?: 'shared' | 'user';
	errorLine?: number;
	errorLineContent?: string;
	errorStack?: string;
	errorFrame?: string;
	errorColumn?: number;
}

type ParsedIdentifier = {
	type: string;
	name?: string;
};

type ParsedDeclaration = {
	id?: ParsedIdentifier;
};

type ParsedStatement = {
	type: string;
	start: number;
	end: number;
	declarations?: ParsedDeclaration[];
};

type ParsedProgram = {
	body?: ParsedStatement[];
};

function getAcornOptions(): AcornOptions {
	return {
		ecmaVersion: 2024,
		sourceType: 'script',
		allowAwaitOutsideFunction: true,
		allowReturnOutsideFunction: true,
		locations: true
	};
}

function parseUserCode(code: string): ParsedProgram {
	return Parser.parse(code ?? '', getAcornOptions()) as ParsedProgram;
}

function prepareCodeForExecution(code: string): string {
	const source = code ?? '';
	const program = parseUserCode(source);
	const body = program.body ?? [];
	const lastStatement = body.at(-1);

	if (!lastStatement) {
		return source;
	}

	if (lastStatement.type === 'ReturnStatement') {
		return source;
	}

	if (lastStatement.type === 'ExpressionStatement') {
		const expressionSource = source.slice(lastStatement.start, lastStatement.end).replace(/;\s*$/, '');
		return `${source.slice(0, lastStatement.start)}return (${expressionSource});${source.slice(lastStatement.end)}`;
	}

	if (lastStatement.type === 'VariableDeclaration') {
		const lastDeclaration = lastStatement.declarations?.at(-1);
		if (lastDeclaration?.id?.type === 'Identifier' && lastDeclaration.id.name) {
			return `${source.slice(0, lastStatement.end)}\nreturn ${lastDeclaration.id.name};${source.slice(lastStatement.end)}`;
		}
	}

	return source;
}

function hasVisibleOutput(result: CodeInterpreterResult): boolean {
	return Boolean(result.error) || result.result !== undefined || (result.logs?.length ?? 0) > 0;
}

function buildNoOutputMessage(code: string): string {
	const source = code ?? '';
	const hasLiteralEscapedNewlines = !/[\r\n]/.test(source) && /\\(?:r|n)/.test(source);
	const startsWithLineComment = /^\s*\/\//.test(source);

	if (hasLiteralEscapedNewlines && startsWithLineComment) {
		return '(no output; this snippet appears to use literal "\\n" escapes instead of real newlines, so the leading // comment may have swallowed the rest of the code. Send raw multi-line code, or add a final expression, explicit return, or console.log.)';
	}

	if (hasLiteralEscapedNewlines) {
		return '(no output; this snippet appears to use literal "\\n" escapes instead of real newlines. Send raw multi-line code, or add a final expression, explicit return, or console.log. If you returned a Promise, await it at top level.)';
	}

	return '(no output; add a final expression, explicit return, or console.log. If you returned a Promise, await it at top level.)';
}

function normalizeSourceForExecution(code: string): string {
	return maybeDecodeEscapedMultilineSource(code) ?? code;
}

function joinSourceBlocks(sharedCode: string, userCode: string): string {
	if (!sharedCode) {
		return userCode;
	}

	if (!userCode) {
		return sharedCode;
	}

	return `${sharedCode}\n${userCode}`;
}

async function runCodeInterpreterOnce(
	code: string,
	timeoutMs = 30_000,
	sharedCode = ''
): Promise<CodeInterpreterResult> {
	let executableCode = code ?? '';
	try {
		executableCode = prepareCodeForExecution(code ?? '');
	} catch (err) {
		const message = err instanceof Error ? err.message : String(err);
		const loc = (err as Error & { loc?: { line: number; column: number } }).loc;
		const userLine = loc?.line ? Math.max(1, loc.line) : undefined;
		const userColumn = typeof loc?.column === 'number' ? loc.column + 1 : undefined;
		const lines = (code ?? '').split('\n');
		const lineContent = userLine ? lines[userLine - 1]?.trim() : undefined;
		return {
			logs: [],
			error: message,
			errorScope: 'user',
			errorLine: userLine,
			errorColumn: userColumn,
			errorLineContent: lineContent
		};
	}

	const normalizedSharedCode = normalizeSourceForExecution(sharedCode ?? '');
	if (normalizedSharedCode) {
		try {
			parseUserCode(normalizedSharedCode);
		} catch (err) {
			const message = err instanceof Error ? err.message : String(err);
			const loc = (err as Error & { loc?: { line: number; column: number } }).loc;
			const sharedLine = loc?.line ? Math.max(1, loc.line) : undefined;
			const sharedColumn = typeof loc?.column === 'number' ? loc.column + 1 : undefined;
			const lines = normalizedSharedCode.split('\n');
			const lineContent = sharedLine ? lines[sharedLine - 1]?.trim() : undefined;
			return {
				logs: [],
				error: message,
				errorScope: 'shared',
				errorLine: sharedLine,
				errorColumn: sharedColumn,
				errorLineContent: lineContent
			};
		}
	}

	const rawCode = joinSourceBlocks(normalizedSharedCode, code ?? '');
	const combinedExecutableCode = joinSourceBlocks(normalizedSharedCode, executableCode);
	const sharedLineCount = normalizedSharedCode ? normalizedSharedCode.split('\n').length : 0;

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

      const run = async (rawCode, executableCode, rawUserCode, rawSharedCode, sharedLineCount) => {
        try {
          const markerStart = '__USER_CODE_START__';
          const markerEnd = '__USER_CODE_END__';
          // Use eval within an async IIFE so we can use return and get better syntax error locations.
          // Lines before user code in wrapped string:
          // 1: (async () => {
          // 2: "use strict";
          // 3: // __USER_CODE_START__
          const USER_OFFSET = 3;
          const sourceName = 'code_interpreter_javascript_execute';
          const wrapped =
            \`(async () => {\\n\` +
            \`"use strict";\\n\` +
            \`// \${markerStart}\\n\` +
            \`\${executableCode ?? ''}\\n\` +
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
          let scope = undefined;
          const userLines = (rawUserCode ?? '').split('\\n');
          const sharedLines = (rawSharedCode ?? '').split('\\n');
          const resolveCombinedLine = (combinedLine) => {
            if (sharedLineCount > 0 && combinedLine <= sharedLineCount) {
              scope = 'shared';
              lineNum = combinedLine;
              lineText = sharedLines[combinedLine - 1]?.trim();
              return;
            }

            scope = 'user';
            const userLine = Math.max(1, combinedLine - sharedLineCount);
            lineNum = userLine;
            lineText = userLines[userLine - 1]?.trim();
          };
          try {
            const stack = String(err?.stack ?? '');
            const match =
              stack.match(/(?:<anonymous>|code_interpreter_javascript_execute):(\\d+):(\\d+)/) ||
              stack.match(/:(\\d+):(\\d+)/); // fallback: first frame with line/col
            if (match) {
              const rawLine = Number(match[1]);
              const rawCol = Number(match[2]);
              const combinedLine = Math.max(1, rawLine - USER_OFFSET);
              resolveCombinedLine(combinedLine);
              columnNum = rawCol;
            }
          } catch {}
          if (!lineNum && err?.message) {
            const idMatch = String(err.message).match(/['"]?([A-Za-z_$][\\w$]*)['"]? is not defined/);
            if (idMatch) {
              const ident = idMatch[1];
              const userIdx = userLines.findIndex((l) => l.includes(ident));
              if (userIdx !== -1) {
                scope = 'user';
                lineNum = userIdx + 1;
                lineText = userLines[userIdx]?.trim();
              } else {
                const sharedIdx = sharedLines.findIndex((l) => l.includes(ident));
                if (sharedIdx !== -1) {
                  scope = 'shared';
                  lineNum = sharedIdx + 1;
                  lineText = sharedLines[sharedIdx]?.trim();
                }
              }
            }
          }
          if (!lineNum) {
            const ln = err?.lineNumber ?? err?.lineno ?? err?.line;
            if (typeof ln === 'number') {
              resolveCombinedLine(Math.max(1, ln));
            }
          }
          if (columnNum === undefined) {
            const col = err?.columnNumber ?? err?.colno ?? undefined;
            if (typeof col === 'number') columnNum = col;
          }
          const stack = err?.stack ? String(err.stack) : undefined;
          const firstStackFrame = stack
            ?.split('\\n')
            .find((l) => l.includes('<anonymous>') || l.includes('code_interpreter_javascript_execute'));
          send({
            type: 'error',
            message: err?.message ?? String(err),
            stack,
            frame: firstStackFrame,
            logs,
            scope,
            line: lineNum,
            lineContent: lineText,
            column: columnNum
          });
        }
      };

      self.onmessage = (e) => {
        run(
          e.data?.rawCode ?? '',
          e.data?.executableCode ?? e.data?.rawCode ?? '',
          e.data?.userCode ?? '',
          e.data?.sharedCode ?? '',
          e.data?.sharedLineCount ?? 0
        );
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
					errorScope: event.data.scope,
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

		worker.postMessage({
			rawCode,
			executableCode: combinedExecutableCode,
			userCode: code ?? '',
			sharedCode: normalizedSharedCode,
			sharedLineCount
		});
	});
}

export async function runCodeInterpreter(
	code: string,
	timeoutMs = 30_000,
	sharedCode = ''
): Promise<CodeInterpreterResult> {
	const initialResult = await runCodeInterpreterOnce(code, timeoutMs, sharedCode);
	if (hasVisibleOutput(initialResult)) {
		return initialResult;
	}

	const decodedCode = maybeDecodeEscapedMultilineSource(code);
	if (!decodedCode) {
		return initialResult;
	}

	const decodedResult = await runCodeInterpreterOnce(decodedCode, timeoutMs, sharedCode);
	return hasVisibleOutput(decodedResult) ? decodedResult : initialResult;
}

function extractCodeArgument(argsJson: string, toolName: string): string {
	return getJavaScriptSourceArgument(toolName, argsJson) ?? argsJson;
}

function getTimeoutMs(config?: Record<string, unknown>): number {
	const timeoutSecondsRaw = config?.codeInterpreterTimeoutSeconds;
	const timeoutSeconds =
		typeof timeoutSecondsRaw === 'number'
			? timeoutSecondsRaw
			: typeof timeoutSecondsRaw === 'string'
				? Number(timeoutSecondsRaw)
				: 30;

	return Number.isFinite(timeoutSeconds)
		? Math.max(0, Math.round(timeoutSeconds * 1000))
		: 30_000;
}

function buildErrorContent(result: CodeInterpreterResult): string {
	const lineLabel = result.errorLine !== undefined ? `line ${result.errorLine}` : null;
	const columnLabel =
		result.errorLine !== undefined && typeof result.errorColumn === 'number'
			? `, col ${result.errorColumn}`
			: '';
	const lineSnippet =
		result.errorLine !== undefined && result.errorLineContent
			? `: ${result.errorLineContent.trim()}`
			: '';
	const lineInfo = lineLabel ? ` (${lineLabel}${columnLabel}${lineSnippet})` : '';
	const errorPrefix =
		result.errorScope === 'shared' ? 'Error in shared JavaScript state' : 'Error';

	let content = `${errorPrefix}${lineInfo}: ${result.error}`;
	if (!lineLabel) {
		if (result.errorFrame) {
			content += `\nFrame: ${result.errorFrame}`;
		} else if (result.errorStack) {
			content += `\nStack: ${result.errorStack}`;
		}
	}

	return content;
}

async function executeCodeInterpreterTool(
	argsJson: string,
	config?: Record<string, unknown>,
	context?: BuiltInToolExecutionContext
): Promise<{ content: string; isError: boolean }> {
	const code = extractCodeArgument(argsJson, CODE_INTERPRETER_JS_TOOL_NAME);
	const sharedCode = buildSharedJavaScriptContext(context?.messages) ?? '';
	const {
		result,
		logs,
		error,
		errorScope,
		errorLine,
		errorLineContent,
		errorStack,
		errorFrame,
		errorColumn
	} = await runCodeInterpreter(code, getTimeoutMs(config), sharedCode);

	let combined = '';
	if (logs?.length) {
		combined += logs.join('\n');
	}

	if (combined && (result !== undefined || error)) {
		combined += '\n';
	}

	if (error) {
		combined += buildErrorContent({
			logs,
			error,
			errorScope,
			errorLine,
			errorLineContent,
			errorStack,
			errorFrame,
			errorColumn
		});
		return { content: combined, isError: true };
	}

	if (result !== undefined) {
		combined += result;
		return { content: combined, isError: false };
	}

	return { content: combined || buildNoOutputMessage(code), isError: false };
}

async function executeCodeContextDefineTool(argsJson: string): Promise<{
	content: string;
	isError: boolean;
}> {
	const code = normalizeSourceForExecution(
		extractCodeArgument(argsJson, CODE_CONTEXT_DEFINE_JS_TOOL_NAME)
	);
	if (!code.trim()) {
		return {
			content: 'Error: shared JavaScript state cannot be empty. Use code_interpreter_javascript_clear_context to clear it.',
			isError: true
		};
	}

	try {
		parseUserCode(code);
	} catch (err) {
		const message = err instanceof Error ? err.message : String(err);
		const loc = (err as Error & { loc?: { line: number; column: number } }).loc;
		const line = loc?.line ? Math.max(1, loc.line) : undefined;
		const column = typeof loc?.column === 'number' ? loc.column + 1 : undefined;
		const lineContent = line ? code.split('\n')[line - 1]?.trim() : undefined;
		const lineInfo = line
			? ` (line ${line}${column !== undefined ? `, col ${column}` : ''}${lineContent ? `: ${lineContent}` : ''})`
			: '';
		return { content: `Error${lineInfo}: ${message}`, isError: true };
	}

	return { content: 'Shared JavaScript state updated.', isError: false };
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
			type: SettingsFieldType.INPUT,
			defaultValue: 30,
			help: 'Maximum time allowed for the JavaScript tool to run before it is terminated.'
		} satisfies ToolSettingDefinition
	],
	definition: codeInterpreterToolDefinition,
	execute: executeCodeInterpreterTool
});

registerTool({
	name: CODE_CONTEXT_DEFINE_JS_TOOL_NAME,
	label: 'Code Interpreter (JavaScript)',
	description: 'Run JavaScript in a sandboxed Worker and capture logs plus final value.',
	enableConfigKey: 'enableCodeInterpreterTool',
	defaultEnabled: true,
	definition: codeContextDefineToolDefinition,
	execute: executeCodeContextDefineTool
});

registerTool({
	name: CODE_CONTEXT_CLEAR_JS_TOOL_NAME,
	label: 'Code Interpreter (JavaScript)',
	description: 'Run JavaScript in a sandboxed Worker and capture logs plus final value.',
	enableConfigKey: 'enableCodeInterpreterTool',
	defaultEnabled: true,
	definition: codeContextClearToolDefinition,
	execute: async () => ({ content: 'Shared JavaScript state cleared.', isError: false })
});
