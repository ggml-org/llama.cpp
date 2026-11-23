// Plain-language lines for tool calls: what one call did ("Read a file
// sheet_new.jpg"), what it is doing now ("Reading package.json") and what
// a run of calls did ("Ran 3 commands, read 2 files"). A call that has the
// arg the settings pick (`description` by default) shows it as the model
// wrote it. Otherwise built-in tools use TOOL_CALL_PHRASING, and MCP and
// unknown tools read as "Used <tool>" with their main argument.

import { extractToolArgString, fileNameFromPath } from './_shared';
import {
	TOOL_ARG_PATH_KEYS,
	TOOL_CALL_MAIN_ARG_KEYS,
	TOOL_CALL_PHRASING,
	TOOL_CALL_SHOWN_ARG_DEFAULT,
	TOOL_CALL_SIMPLE_ARG_MAX_CHARS,
	TOOL_CALL_SUMMARY,
	TOOL_CALL_TARGET_MAX_CHARS,
	TOOL_CALL_THINKING_LABEL,
	TOOL_SEARCH_LOAD_PHRASING,
	TOOL_SEARCH_PHRASING_NAME,
	TOOL_SEARCH_PHRASING_SELECT_PREFIX,
	TOOL_SEARCH_QUERY_PHRASING,
	WEB_SEARCH_PHRASING
} from '#lib/constants/index.ts';
import { AgenticSectionType, BuiltInTool, ToolCallSummaryKind } from '#lib/enums/index.ts';
import type { AgenticSection, ToolCallDescription, ToolCallPhrasing } from '#lib/types/index.ts';
import { isWebSearchToolName, memoizeBounded, parsePartialJsonArgs } from '#lib/utils/index.ts';

/** How many results each cache below holds. */
const CACHE_MAX_SIZE = 256;
/** Args longer than this ( whole files ) are not worth holding as cache keys. */
const CACHE_MAX_ARGS_CHARS = 2000;

type ToolArgField = { label: string; value: string };

function shorten(text: string): string {
	const oneLine = text.replace(/\s+/g, ' ').trim();

	return oneLine.length > TOOL_CALL_TARGET_MAX_CHARS
		? `${oneLine.slice(0, TOOL_CALL_TARGET_MAX_CHARS - 3)}...`
		: oneLine;
}

function stringArg(args: Record<string, unknown> | null, key: string): string | undefined {
	const value = args?.[key];

	return typeof value === 'string' && value.trim() ? value : undefined;
}

function genericMainArg(args: Record<string, unknown> | null): string | undefined {
	if (!args) return undefined;

	for (const key of TOOL_CALL_MAIN_ARG_KEYS) {
		const value = stringArg(args, key);

		if (value) return value;
	}

	for (const value of Object.values(args)) {
		if (typeof value === 'number') return String(value);

		if (typeof value === 'string' && value.trim() && value.length <= TOOL_CALL_TARGET_MAX_CHARS) {
			return value;
		}
	}

	return undefined;
}

function genericPhrasing(toolName: string): ToolCallPhrasing {
	return {
		object: toolName,
		past: 'Used',
		present: 'Using',
		summary: ToolCallSummaryKind.OTHER,
		targetObject: toolName
	};
}

interface ResolvedCall {
	phrasing: ToolCallPhrasing;
	target?: string;
	count?: number;
}

function resolveCall(toolName: string, toolArgs: string): ResolvedCall {
	// write_file and edit_file args hold whole files: read the path with a
	// targeted match instead of parsing the blob
	if (
		toolName === BuiltInTool.SERVER_READ_FILE ||
		toolName === BuiltInTool.SERVER_WRITE_FILE ||
		toolName === BuiltInTool.SERVER_EDIT_FILE ||
		toolName === BuiltInTool.BROWSER_READ_MEDIA
	) {
		const path =
			extractToolArgString(toolArgs, TOOL_ARG_PATH_KEYS) ??
			stringArg(parsePartialJsonArgs(toolArgs), 'path');

		return { phrasing: TOOL_CALL_PHRASING[toolName], target: fileNameFromPath(path) };
	}

	const args = toolArgs ? parsePartialJsonArgs(toolArgs) : null;

	if (toolName === BuiltInTool.SERVER_EXEC_SHELL_COMMAND) {
		// the targeted match keeps the command in the line while a later arg
		// is still streaming and the JSON does not parse
		return {
			phrasing: TOOL_CALL_PHRASING[toolName],
			target: stringArg(args, 'command') ?? extractToolArgString(toolArgs, ['command'])
		};
	}

	switch (toolName) {
		case BuiltInTool.SERVER_GREP_SEARCH:
			return { phrasing: TOOL_CALL_PHRASING[toolName], target: stringArg(args, 'pattern') };

		case BuiltInTool.SERVER_FILE_GLOB_SEARCH: {
			const include = stringArg(args, 'include');

			return {
				phrasing: TOOL_CALL_PHRASING[toolName],
				target: include && include !== '**' ? include : undefined
			};
		}

		case BuiltInTool.BROWSER_SEARCH_CHATS: {
			const matches = Array.isArray(args?.by_matches)
				? args.by_matches.filter((m): m is string => typeof m === 'string' && m.trim() !== '')
				: [];

			return { phrasing: TOOL_CALL_PHRASING[toolName], target: matches.join(', ') || undefined };
		}

		case TOOL_SEARCH_PHRASING_NAME: {
			const query = stringArg(args, 'query')?.trim();

			if (query?.startsWith(TOOL_SEARCH_PHRASING_SELECT_PREFIX)) {
				const names = query
					.slice(TOOL_SEARCH_PHRASING_SELECT_PREFIX.length)
					.split(',')
					.map((name) => name.trim())
					.filter(Boolean);

				return {
					count: Math.max(names.length, 1),
					phrasing: TOOL_SEARCH_LOAD_PHRASING,
					target: names.join(', ') || undefined
				};
			}

			return { phrasing: TOOL_SEARCH_QUERY_PHRASING, target: query };
		}
	}

	if (isWebSearchToolName(toolName)) {
		return { phrasing: WEB_SEARCH_PHRASING, target: stringArg(args, 'query') };
	}

	const known = TOOL_CALL_PHRASING[toolName];

	if (known) return { phrasing: known };

	return { phrasing: genericPhrasing(toolName), target: genericMainArg(args) };
}

/** An arg name the settings can pick: a plain key, so it is safe in a pattern. */
const SHOWN_ARG_PATTERN = /^[\w-]+$/;

/**
 * The value of the arg a call's line shows, when the call has it. Whole-file
 * args are not parsed for it: the targeted match reads it instead.
 */
function shownArgValue(toolArgs: string, shownArg: string | null): string | undefined {
	if (!shownArg || !toolArgs || !SHOWN_ARG_PATTERN.test(shownArg)) return undefined;

	const value =
		toolArgs.length <= CACHE_MAX_ARGS_CHARS
			? stringArg(parsePartialJsonArgs(toolArgs), shownArg)
			: extractToolArgString(toolArgs, [shownArg]);

	return value?.trim() || undefined;
}

/**
 * Describe one tool call as a short sentence, in the past tense once it is
 * done and the present tense while `running`. `shownArg` names the arg
 * shown instead when the call has it ( null for none ).
 */
export function describeToolCall(
	toolName: string | undefined,
	toolArgs: string | undefined,
	running = false,
	shownArg: string | null = TOOL_CALL_SHOWN_ARG_DEFAULT
): ToolCallDescription {
	return describeCached(toolName || 'a tool', toolArgs ?? '', running, shownArg ?? '');
}

// Sections are rebuilt on every streamed chunk, so the same finished call is
// described again and again: remember the answer for its args. Every caller
// of one call reads the same frozen object, so a title cannot edit the line
// out for the others.
const describeCached = memoizeBounded(
	(name: string, args: string, running: boolean, shownArg: string) =>
		Object.freeze(buildDescription(name, args, running, shownArg || null)),
	(name, args, running, shownArg) =>
		args.length <= CACHE_MAX_ARGS_CHARS
			? `${running ? 1 : 0}\0${shownArg}\0${name}\0${args}`
			: null,
	CACHE_MAX_SIZE
);

function buildDescription(
	name: string,
	args: string,
	running: boolean,
	shownArg: string | null
): ToolCallDescription {
	const { count = 1, phrasing, target } = resolveCall(name, args);
	const shown = shownArgValue(args, shownArg);
	// an arg the row already shows as its target ( a shell command ) would
	// only repeat it
	const described = shown && shown !== target?.trim() ? shown : undefined;

	if (described) {
		return {
			count,
			described: true,
			// shown as the model wrote it: rewriting its tense by rule mangles
			// anything that does not start with a plain verb
			label: shorten(described).replace(/[.\s]+$/, ''),
			quoted: false,
			summary: phrasing.summary
		};
	}

	const verb = running ? phrasing.present : phrasing.past;
	const shortTarget = target ? shorten(target) : undefined;
	const object = shortTarget ? phrasing.targetObject : phrasing.object;

	return {
		count,
		label: object ? `${verb} ${object}` : verb,
		quoted: Boolean(phrasing.quoteTarget && shortTarget),
		summary: phrasing.summary,
		target: shortTarget
	};
}

/** One line of plain text for a description, e.g. `Searched for 'issue'`. */
export function formatToolCallLine(description: ToolCallDescription): string {
	if (!description.target) return description.label;

	const target = description.quoted ? `'${description.target}'` : description.target;

	return `${description.label} ${target}`;
}

export function isToolCallSection(section: AgenticSection): boolean {
	return (
		section.type === AgenticSectionType.TOOL_CALL ||
		section.type === AgenticSectionType.TOOL_CALL_PENDING ||
		section.type === AgenticSectionType.TOOL_CALL_STREAMING
	);
}

/** True while the call has no result yet and the reply is still streaming. */
export function isToolCallRunning(section: AgenticSection, isStreaming: boolean): boolean {
	return (
		isStreaming &&
		(section.type === AgenticSectionType.TOOL_CALL_PENDING ||
			section.type === AgenticSectionType.TOOL_CALL_STREAMING)
	);
}

/**
 * Summarize a run of tool calls, e.g. "Ran 3 commands, read 2 files".
 * Kinds keep the order they first appear in; other tools come last.
 */
export function summarizeToolCalls(sections: readonly AgenticSection[]): string {
	const counts = new Map<ToolCallSummaryKind, number>();

	for (const section of sections) {
		if (!isToolCallSection(section)) continue;

		const { count, summary } = describeToolCall(section.toolName, section.toolArgs);

		counts.set(summary, (counts.get(summary) ?? 0) + count);
	}

	const other = counts.get(ToolCallSummaryKind.OTHER);

	if (other !== undefined) {
		counts.delete(ToolCallSummaryKind.OTHER);
		counts.set(ToolCallSummaryKind.OTHER, other);
	}

	const clauses = [...counts].map(([kind, n]) => TOOL_CALL_SUMMARY[kind](n));
	const text = clauses.join(', ');

	return text.charAt(0).toUpperCase() + text.slice(1);
}

/**
 * What a run of calls is doing now, in user words ("Running a command",
 * "Thinking"): the model thinking at its end, or the last call still
 * running; null when neither is.
 */
export function describeToolActivity(
	sections: readonly AgenticSection[],
	isStreaming: boolean,
	executingToolCallId?: string | null,
	shownArgFor: (toolName: string | undefined) => string | null = () => TOOL_CALL_SHOWN_ARG_DEFAULT
): string | null {
	// the model thinking at the end of the run, between calls or before its answer
	if (isStreaming && sections.at(-1)?.type === AgenticSectionType.REASONING_PENDING) {
		return TOOL_CALL_THINKING_LABEL;
	}

	for (let i = sections.length - 1; i >= 0; i--) {
		const section = sections[i];
		const executing =
			section.toolCallId !== undefined && section.toolCallId === executingToolCallId;

		if (executing || isToolCallRunning(section, isStreaming)) {
			return formatToolCallLine(
				describeToolCall(section.toolName, section.toolArgs, true, shownArgFor(section.toolName))
			);
		}
	}

	return null;
}

/**
 * Tool args as labelled fields when every value is a short, one-line
 * scalar; null when the args need the JSON view instead. Memoized like
 * describeToolCall: the row that shows them re-renders on every chunk.
 */
export function simpleToolArgFields(toolArgs: string | undefined): ToolArgField[] | null {
	return toolArgs ? argFieldsCached(toolArgs) : null;
}

const argFieldsCached = memoizeBounded(
	buildSimpleToolArgFields,
	(toolArgs) => (toolArgs.length <= CACHE_MAX_ARGS_CHARS ? toolArgs : null),
	CACHE_MAX_SIZE
);

function buildSimpleToolArgFields(toolArgs: string): ToolArgField[] | null {
	let parsed: unknown;

	try {
		parsed = JSON.parse(toolArgs);
	} catch {
		return null;
	}

	if (!parsed || typeof parsed !== 'object' || Array.isArray(parsed)) return null;

	const entries = Object.entries(parsed as Record<string, unknown>);

	if (entries.length === 0) return null;

	const fields: ToolArgField[] = [];

	for (const [key, value] of entries) {
		if (value !== null && typeof value === 'object') return null;

		const text = String(value);

		if (text.length > TOOL_CALL_SIMPLE_ARG_MAX_CHARS || text.includes('\n')) return null;

		fields.push({ label: humanizeArgKey(key), value: text });
	}

	return fields;
}

/** `max_results` / `maxResults` -> `Max results`. */
export function humanizeArgKey(key: string): string {
	const words = key
		.replace(/([a-z0-9])([A-Z])/g, '$1 $2')
		.replace(/[_-]+/g, ' ')
		.trim()
		.toLowerCase();

	return words.charAt(0).toUpperCase() + words.slice(1);
}

export function isReasoningSection(section: AgenticSection): boolean {
	return (
		section.type === AgenticSectionType.REASONING ||
		section.type === AgenticSectionType.REASONING_PENDING
	);
}

/**
 * Split section positions into single items and runs of tool work, e.g.
 * [0, [1, 2, 3], 4]. A run is the calls and all the reasoning around them,
 * up to the next text, so the answer follows the work as one row: it is
 * grouped once it holds `minCalls` calls, or one call with reasoning beside
 * it. Reasoning with no call in reach, and text, stay apart.
 */
export function groupToolCallRuns(
	sections: readonly AgenticSection[],
	indices: readonly number[],
	minCalls: number
): (number | number[])[] {
	const items: (number | number[])[] = [];

	let run: number[] = [];
	let calls = 0;
	// reasoning seen since the last call: it ends the run if text follows,
	// and leads into it if another call does
	let thinking: number[] = [];

	const flush = () => {
		const whole = [...run, ...thinking];

		if (calls >= minCalls || (calls > 0 && whole.length > calls)) items.push(whole);
		else items.push(...whole);

		run = [];
		thinking = [];
		calls = 0;
	};

	for (const index of indices) {
		const section = sections[index];

		if (isToolCallSection(section)) {
			run.push(...thinking, index);
			thinking = [];
			calls++;
		} else if (isReasoningSection(section)) {
			thinking.push(index);
		} else {
			flush();
			items.push(index);
		}
	}

	flush();

	return items;
}
