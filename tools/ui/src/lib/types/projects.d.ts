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
	/** Put away: off the sidebar and the pickers, with its chats, until restored. */
	archived?: boolean;
	contextItems: ProjectContextItem[];
	createdAt: number;
	/** Working directory new conversations in this project start in. */
	cwd: string | null;
	/** Answer to "What are you trying to achieve?" */
	description: string;
	/** Sidebar group the project sits in, like a chat's. */
	groupId?: string;
	id: string;
	instructions: string;
	name: string;
	/** Shown in Pinned, wherever its group is. */
	pinned?: boolean;
	updatedAt: number;
}

/** What the project dialog edits: everything but the bookkeeping. */
export type ProjectDraft = Pick<
	Project,
	'contextItems' | 'cwd' | 'description' | 'instructions' | 'name'
>;
