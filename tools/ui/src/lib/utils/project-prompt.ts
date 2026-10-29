/**
 * System prompt section for a project, so every chat started in it sees the
 * same instructions and the documents the user referenced.
 */

import { ProjectContextKind } from '$lib/enums';
import type { Project } from '$lib/types';

/** Rough characters per token, enough for a budget hint without a tokenizer. */
const CHARS_PER_TOKEN = 4;

/** The parts of a project, saved or still a draft, that its chats start with. */
type ProjectPromptSource = Pick<Project, 'contextItems' | 'instructions' | 'name'>;

export function buildProjectSystemPrompt(project: ProjectPromptSource | undefined): string {
	if (!project) return '';

	const sections: string[] = [];
	const instructions = project.instructions.trim();

	if (instructions) {
		sections.push(`Instructions for this project:\n${instructions}`);
	}

	if (project.contextItems.length > 0) {
		const items = project.contextItems.map((item) => {
			if (item.kind === ProjectContextKind.MCP_RESOURCE) {
				return `- ${item.name} (MCP resource ${item.path ?? ''} on ${item.serverName ?? 'server'})`;
			}

			if (item.kind === ProjectContextKind.FILE) return `- ${item.path ?? item.name} (server file)`;

			return `- ${item.name} (uploaded file)`;
		});

		sections.push(`Context the user added to this project:\n${items.join('\n')}`);
	}

	if (sections.length === 0) return '';

	return `You are working in the project "${project.name}".\n\n${sections.join('\n\n')}`;
}

/**
 * Tokens every chat in the project starts with, and how many context items
 * are only read when the model asks for them.
 */
export function estimateProjectContext(project: ProjectPromptSource): {
	readOnDemand: number;
	tokens: number;
} {
	const uploadedText = project.contextItems
		.map((item) =>
			item.kind === ProjectContextKind.ATTACHMENT && item.extra && 'content' in item.extra
				? item.extra.content
				: ''
		)
		.join('');
	const chars = buildProjectSystemPrompt(project).length + uploadedText.length;

	return {
		readOnDemand: project.contextItems.filter((item) => item.kind !== ProjectContextKind.ATTACHMENT)
			.length,
		tokens: Math.round(chars / CHARS_PER_TOKEN)
	};
}
