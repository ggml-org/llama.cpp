import type { AgenticQuestionPrompt } from '$lib/types';

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

	return {
		question,
		header:
			typeof record.header === 'string' && record.header.trim() ? record.header.trim() : question,
		options,
		multiple: record.multiple === true,
		custom: record.custom !== false
	};
}

export function parseQuestionToolArguments(
	args: string | Record<string, unknown> | undefined
): AgenticQuestionPrompt[] {
	if (!args) return [];

	let parsed: Record<string, unknown> | null = null;
	if (typeof args === 'string') {
		try {
			parsed = asRecord(JSON.parse(args));
		} catch {
			return [];
		}
	} else {
		parsed = args;
	}

	if (!parsed || !Array.isArray(parsed.questions)) return [];
	return parsed.questions
		.map(normalizePrompt)
		.filter((question): question is AgenticQuestionPrompt => Boolean(question));
}

export function parseQuestionToolResult(
	result: string | undefined,
	questions: AgenticQuestionPrompt[]
): AgenticQuestionAnswerPair[] {
	if (
		!result?.startsWith(ANSWER_RESULT_PREFIX) ||
		!result.endsWith(ANSWER_RESULT_SUFFIX) ||
		questions.length === 0
	) {
		return [];
	}

	const body = result.slice(ANSWER_RESULT_PREFIX.length, -ANSWER_RESULT_SUFFIX.length);
	const pairs: AgenticQuestionAnswerPair[] = [];
	let offset = 0;

	for (let index = 0; index < questions.length; index += 1) {
		const question = questions[index];
		const marker = `"${question.question}"="`;
		const markerStart = body.indexOf(marker, offset);
		if (markerStart === -1) return [];

		const answerStart = markerStart + marker.length;
		const nextQuestion = questions[index + 1];
		const answerEnd = nextQuestion
			? body.indexOf(`", "${nextQuestion.question}"="`, answerStart)
			: body.endsWith('"')
				? body.length - 1
				: -1;
		if (answerEnd === -1) return [];

		pairs.push({
			question: question.question,
			answer: body.slice(answerStart, answerEnd).trim()
		});
		offset = answerEnd + 3;
	}

	return pairs;
}
