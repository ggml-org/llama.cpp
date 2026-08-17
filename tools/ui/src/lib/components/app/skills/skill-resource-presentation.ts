import { isTextFileByName } from '$lib/utils';

export type SkillResourceFormat = 'markdown' | 'html' | 'source' | 'unsupported';

export interface SkillResourceFileNode {
	kind: 'file';
	name: string;
	path: string;
	format: SkillResourceFormat;
}

export interface SkillResourceFolderNode {
	kind: 'folder';
	name: string;
	path: string;
	children: SkillResourceTreeNode[];
}

export type SkillResourceTreeNode = SkillResourceFileNode | SkillResourceFolderNode;

export interface SkillResourceTreeRow {
	node: SkillResourceTreeNode;
	depth: number;
	parentPath: string | null;
}

export function classifySkillResourceFormat(path: string): SkillResourceFormat {
	const lowerPath = path.toLowerCase();

	if (lowerPath === 'skill.md' || lowerPath.endsWith('.md') || lowerPath.endsWith('.markdown')) {
		return 'markdown';
	}

	if (lowerPath.endsWith('.html') || lowerPath.endsWith('.htm')) {
		return 'html';
	}

	// FileExtensionText currently contains ZIP for upload handling. A skill archive is not text.
	if (lowerPath.endsWith('.zip')) {
		return 'unsupported';
	}

	return isTextFileByName(path) ? 'source' : 'unsupported';
}

export function createSkillRootNode(): SkillResourceFileNode {
	return {
		format: 'markdown',
		kind: 'file',
		name: 'SKILL.md',
		path: 'SKILL.md'
	};
}

export function buildSkillResourceTree(paths: readonly string[]): SkillResourceTreeNode[] {
	const roots: SkillResourceTreeNode[] = [];
	const folders = new Map<string, SkillResourceFolderNode>();
	const files = new Set<string>();

	for (const path of paths) {
		if (files.has(path)) continue;

		const segments = path.split('/');

		if (segments.length === 0 || segments.some((segment) => segment.length === 0)) continue;

		let children = roots;
		let parentPath = '';

		for (const segment of segments.slice(0, -1)) {
			const folderPath = parentPath ? `${parentPath}/${segment}` : segment;
			let folder = folders.get(folderPath);

			if (!folder) {
				folder = { children: [], kind: 'folder', name: segment, path: folderPath };
				folders.set(folderPath, folder);
				children.push(folder);
			}

			children = folder.children;
			parentPath = folderPath;
		}

		const name = segments[segments.length - 1];
		children.push({ format: classifySkillResourceFormat(path), kind: 'file', name, path });
		files.add(path);
	}

	return roots;
}

export function getInitialExpandedFolderPaths(
	tree: readonly SkillResourceTreeNode[]
): ReadonlySet<string> {
	return new Set(
		tree.filter((node): node is SkillResourceFolderNode => node.kind === 'folder').map((node) => node.path)
	);
}

export function flattenSkillResourceTree(
	tree: readonly SkillResourceTreeNode[],
	expandedPaths: ReadonlySet<string>
): SkillResourceTreeRow[] {
	const rows: SkillResourceTreeRow[] = [];

	function append(
		nodes: readonly SkillResourceTreeNode[],
		depth: number,
		parentPath: string | null
	) {
		for (const node of nodes) {
			rows.push({ depth, node, parentPath });

			if (node.kind === 'folder' && expandedPaths.has(node.path)) {
				append(node.children, depth + 1, node.path);
			}
		}
	}

	append(tree, 0, null);

	return rows;
}

export function findSkillResourceParentPath(
	rows: readonly SkillResourceTreeRow[],
	path: string
): string | null {
	return rows.find((row) => row.node.path === path)?.parentPath ?? null;
}
