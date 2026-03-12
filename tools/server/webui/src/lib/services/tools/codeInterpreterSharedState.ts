import type { AgenticMessage, AgenticToolCallPayload } from '$lib/types/agentic';
import { AgenticSectionType } from '$lib/enums';
import { parseAgenticContent } from '$lib/utils/agentic';
import { maybeDecodeEscapedMultilineSource } from './codeInterpreterFormatting';

export const CODE_INTERPRETER_JS_TOOL_NAME = 'code_interpreter_javascript_execute';
export const CODE_CONTEXT_DEFINE_JS_TOOL_NAME = 'code_interpreter_javascript_set_context';
export const CODE_CONTEXT_CLEAR_JS_TOOL_NAME = 'code_interpreter_javascript_clear_context';

const JAVASCRIPT_SOURCE_TOOL_NAMES = new Set([
	CODE_INTERPRETER_JS_TOOL_NAME,
	CODE_CONTEXT_DEFINE_JS_TOOL_NAME
]);

function findJsonStringPropertyValueStart(source: string, propertyName: string): number | undefined {
	const propertyPattern = new RegExp(`"${propertyName}"\\s*:\\s*"`, 'g');
	const match = propertyPattern.exec(source);

	if (!match) {
		return undefined;
	}

	return match.index + match[0].length;
}

function decodeMaybePartialJsonString(source: string, startIndex: number): string {
	let decoded = '';
	let index = startIndex;

	while (index < source.length) {
		const char = source[index];

		if (char === '"') {
			return decoded;
		}

		if (char !== '\\') {
			decoded += char;
			index += 1;
			continue;
		}

		const escaped = source[index + 1];
		if (escaped === undefined) {
			decoded += '\\';
			break;
		}

		switch (escaped) {
			case '"':
				decoded += '"';
				index += 2;
				break;
			case '\\':
				decoded += '\\';
				index += 2;
				break;
			case '/':
				decoded += '/';
				index += 2;
				break;
			case 'b':
				decoded += '\b';
				index += 2;
				break;
			case 'f':
				decoded += '\f';
				index += 2;
				break;
			case 'n':
				decoded += '\n';
				index += 2;
				break;
			case 'r':
				decoded += '\n';
				index += 2;
				break;
			case 't':
				decoded += '\t';
				index += 2;
				break;
			case 'u': {
				const hex = source.slice(index + 2, index + 6);
				if (/^[0-9a-fA-F]{4}$/.test(hex)) {
					decoded += String.fromCharCode(Number.parseInt(hex, 16));
					index += 6;
					break;
				}

				decoded += source.slice(index);
				return decoded;
			}
			default:
				decoded += escaped;
				index += 2;
				break;
		}
	}

	return decoded;
}

function extractStringPropertyFromPossiblyPartialJson(
	source: string,
	propertyName: string
): string | undefined {
	const valueStart = findJsonStringPropertyValueStart(source, propertyName);
	if (valueStart === undefined) {
		return undefined;
	}

	return decodeMaybePartialJsonString(source, valueStart);
}

function normalizeJavaScriptSource(code: string): string {
	return maybeDecodeEscapedMultilineSource(code) ?? code;
}

function extractTextContent(content: AgenticMessage['content']): string {
	if (typeof content === 'string') {
		return content;
	}

	if (!Array.isArray(content)) {
		return '';
	}

	return content
		.map((part) =>
			part && typeof part === 'object' && part.type === 'text' && typeof part.text === 'string'
				? part.text
				: ''
		)
		.filter(Boolean)
		.join('\n');
}

function isErrorText(text: string): boolean {
	return text.trimStart().startsWith('Error:');
}

function isToolErrorMessage(content: AgenticMessage['content']): boolean {
	return isErrorText(extractTextContent(content));
}

function applySharedStateToolResult(
	toolName: string | undefined,
	toolArgs: string | undefined,
	toolResult: string,
	sharedCode: string | undefined
): string | undefined {
	if (!toolName || isErrorText(toolResult)) {
		return sharedCode;
	}

	if (toolName === CODE_CONTEXT_CLEAR_JS_TOOL_NAME) {
		return undefined;
	}

	if (toolName === CODE_CONTEXT_DEFINE_JS_TOOL_NAME && toolArgs) {
		return getJavaScriptSourceArgument(toolName, toolArgs);
	}

	return sharedCode;
}

export function getJavaScriptSourceArgument(
	toolName?: string,
	argsJson?: string
): string | undefined {
	if (!toolName || !argsJson || !JAVASCRIPT_SOURCE_TOOL_NAMES.has(toolName)) {
		return undefined;
	}

	try {
		const parsedArgs = JSON.parse(argsJson) as { code?: unknown };
		if (typeof parsedArgs.code !== 'string') {
			return undefined;
		}

		return normalizeJavaScriptSource(parsedArgs.code);
	} catch {
		const partialCode = extractStringPropertyFromPossiblyPartialJson(argsJson, 'code');
		return partialCode ? normalizeJavaScriptSource(partialCode) : undefined;
	}
}

export function buildSharedJavaScriptContext(messages?: AgenticMessage[]): string | undefined {
	if (!messages?.length) {
		return undefined;
	}

	const pendingToolCalls = new Map<string, AgenticToolCallPayload>();
	let sharedCode: string | undefined;

	for (const message of messages) {
		if (message.role === 'assistant') {
			if (typeof message.content === 'string' && message.content.includes('<<<AGENTIC_TOOL_CALL_START>>>')) {
				for (const section of parseAgenticContent(message.content)) {
					if (
						section.type !== AgenticSectionType.TOOL_CALL ||
						!section.toolResult
					) {
						continue;
					}

					sharedCode = applySharedStateToolResult(
						section.toolName,
						section.toolArgs,
						section.toolResult,
						sharedCode
					);
				}
			}

			for (const toolCall of message.tool_calls ?? []) {
				pendingToolCalls.set(toolCall.id, toolCall);
			}
			continue;
		}

		if (message.role !== 'tool') {
			continue;
		}

		const toolCall = pendingToolCalls.get(message.tool_call_id);
		if (!toolCall) {
			continue;
		}

		pendingToolCalls.delete(message.tool_call_id);

		if (isToolErrorMessage(message.content)) {
			continue;
		}

		sharedCode = applySharedStateToolResult(
			toolCall.function.name,
			toolCall.function.arguments,
			extractTextContent(message.content),
			sharedCode
		);
	}

	return sharedCode;
}
