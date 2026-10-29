import type { ProjectContextKind } from '$lib/enums';
import type { DatabaseMessageExtra } from '$lib/types/database';

/**
 * One piece of shared context of a project: an uploaded file, a file on the
 * server, or an MCP resource. Every chat started in the project sees it.
 */
export interface ProjectContextItem {
	/** Parsed attachment, only on `ATTACHMENT` items. */
	extra?: DatabaseMessageExtra;
	id: string;
	kind: ProjectContextKind;
	/** File name as the user knows it. */
	name: string;
	/** Server path for a file, resource URI for an MCP resource. */
	path?: string;
	/** MCP server owning the resource, on `MCP_RESOURCE` items. */
	serverName?: string;
	size?: number;
}

/** A project groups chats under shared instructions and context. */
export interface Project {
	contextItems: ProjectContextItem[];
	createdAt: number;
	/** Working directory new conversations in this project start in. */
	cwd: string | null;
	/** Answer to "What are you trying to achieve?" */
	description: string;
	/** Off the sidebar, kept with its chats until shown again. */
	hidden: boolean;
	id: string;
	instructions: string;
	name: string;
	updatedAt: number;
}

/** What the project dialog edits: everything but the bookkeeping. */
export type ProjectDraft = Pick<
	Project,
	'contextItems' | 'cwd' | 'description' | 'instructions' | 'name'
>;
