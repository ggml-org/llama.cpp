import type { AgenticQuestionAnswers, AgenticQuestionPrompt, AgenticQuestionType } from '$lib/types/agentic';

const ANSWER_RESULT_PREFIX = 'User has answered your questions: ';
const ANSWER_RESULT_SUFFIX = ". You can now continue with the user's answers in mind.";

export interface AgenticQuestionAnswerPair {
	question: string;
	answer: string;
}

function asRecord(value: unknown): Record<string, unknown> | null {
	if (typeof value !== 'object' || value === null || Array.isArray(value)) return null;
	return value as Record<string, unknown>;
}

function parseArguments(args: string | Record<string, unknown>): Record<string, unknown> {
	if (typeof args === 'object') return args;
	const trimmed = args.trim();
	if (trimmed === '') return {};
	return JSON.parse(trimmed) as Record<string, unknown>;
}

function normalizeOption(value: unknown): { label: string; description: string } | null {
	const record = asRecord(value);
	if (!record) return null;

	const label = typeof record.label === 'string' ? record.label.trim() : '';
	if (!label) return null;

	return {
		label,
		description: typeof record.description === 'string' ? record.description.trim() : ''
	};
}

function normalizeQuestionType(value: unknown): AgenticQuestionType | null {
	if (typeof value !== 'string') return null;

	const normalized = value
		.trim()
		.toLowerCase()
		.replace(/[\s-]+/g, '_');

	switch (normalized) {
		case 'single':
		case 'single_choice':
		case 'radio':
			return 'single_choice';
		case 'multiple':
		case 'multiple_choice':
		case 'multi':
		case 'checkbox':
		case 'checkboxes':
			return 'multiple_choice';
		case 'freeform':
		case 'custom':
		case 'text':
		case 'type_your_own_answer':
			return 'freeform';
		default:
			return null;
	}
}

function normalizePrompt(value: unknown): AgenticQuestionPrompt | null {
	const record = asRecord(value);
	if (!record) return null;

	const question = typeof record.question === 'string' ? record.question.trim() : '';
	if (!question) return null;

	const options = Array.isArray(record.options)
		? record.options
				.map(normalizeOption)
				.filter((option): option is NonNullable<typeof option> => Boolean(option))
		: [];
	const requestedType = normalizeQuestionType(record.type);
	const derivedType = requestedType ?? (record.multiple === true ? 'multiple_choice' : 'single_choice');
	const type = options.length === 0 || derivedType === 'freeform' ? 'freeform' : derivedType;

	return {
		question,
		header:
			typeof record.header === 'string' && record.header.trim() ? record.header.trim() : question,
		options: type === 'freeform' ? [] : options,
		type,
		multiple: type === 'multiple_choice',
		custom: type === 'freeform' ? true : record.custom !== false
	};
}

export function parseQuestionToolArguments(
	args: string | Record<string, unknown>
): AgenticQuestionPrompt[] {
	const parsed = parseArguments(args);
	const questions = parsed.questions;

	if (!Array.isArray(questions)) throw new Error('question tool requires a questions array');

	const normalized = questions
		.map(normalizePrompt)
		.filter((question): question is AgenticQuestionPrompt => Boolean(question));

	if (normalized.length === 0) throw new Error('question tool requires at least one valid question');

	return normalized;
}

export function parseQuestionToolResult(result: string): AgenticQuestionAnswerPair[] {
	if (!result.startsWith(ANSWER_RESULT_PREFIX)) return [];

	const body = result.slice(ANSWER_RESULT_PREFIX.length).replace(ANSWER_RESULT_SUFFIX, '');
	const pairs: AgenticQuestionAnswerPair[] = [];
	const pairRegex = /"([^"]+)"="([^"]*)"/g;

	for (const match of body.matchAll(pairRegex)) {
		const question = match[1]?.trim() ?? '';
		const answer = match[2]?.trim() ?? '';
		if (question) pairs.push({ question, answer });
	}

	return pairs;
}

export function formatQuestionToolResult(
	questions: AgenticQuestionPrompt[],
	answers: AgenticQuestionAnswers
): string {
	const formattedAnswers = questions
		.map((question, index) => {
			const answer = answers[index]?.filter(Boolean).join(', ') ?? '';
			return `"${question.question}"="${answer}"`;
		})
		.join(', ');

	return `${ANSWER_RESULT_PREFIX}${formattedAnswers}${ANSWER_RESULT_SUFFIX}`;
}
