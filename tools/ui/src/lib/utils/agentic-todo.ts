import type { AgenticTodoItem, AgenticTodoStatus } from '$lib/types/agentic';

const VALID_STATUSES = new Set<AgenticTodoStatus>([
	'pending',
	'in_progress',
	'completed',
	'cancelled'
]);

function parseArguments(args: string | Record<string, unknown>): Record<string, unknown> {
	if (typeof args === 'object') return args;
	const trimmed = args.trim();
	if (trimmed === '') return {};
	return JSON.parse(trimmed) as Record<string, unknown>;
}

function asRecord(value: unknown): Record<string, unknown> | null {
	if (typeof value !== 'object' || value === null || Array.isArray(value)) return null;
	return value as Record<string, unknown>;
}

function normalizeStatus(value: unknown): AgenticTodoStatus {
	return typeof value === 'string' && VALID_STATUSES.has(value as AgenticTodoStatus)
		? (value as AgenticTodoStatus)
		: 'pending';
}

function normalizeTodo(value: unknown): AgenticTodoItem | null {
	const record = asRecord(value);
	if (!record) return null;

	const content = typeof record.content === 'string' ? record.content.trim() : '';
	if (!content) return null;

	return {
		content,
		status: normalizeStatus(record.status)
	};
}

export function parseTodoWriteArguments(args: string | Record<string, unknown>): AgenticTodoItem[] {
	const parsed = parseArguments(args);

	if (!Array.isArray(parsed.todos)) throw new Error('todowrite tool requires a todos array');

	const todos = parsed.todos
		.map(normalizeTodo)
		.filter((todo): todo is AgenticTodoItem => Boolean(todo));

	if (todos.length === 0) throw new Error('todowrite tool requires at least one valid todo');

	return todos;
}

export function parseTodoWriteResult(result: string | unknown[]): AgenticTodoItem[] {
	let parsed: unknown;

	try {
		parsed = typeof result === 'string' ? JSON.parse(result) : result;
	} catch {
		return [];
	}

	if (!Array.isArray(parsed)) return [];

	return parsed.map(normalizeTodo).filter((todo): todo is AgenticTodoItem => Boolean(todo));
}

export function formatTodoWriteResult(todos: AgenticTodoItem[]): string {
	return JSON.stringify(todos, null, 2);
}
