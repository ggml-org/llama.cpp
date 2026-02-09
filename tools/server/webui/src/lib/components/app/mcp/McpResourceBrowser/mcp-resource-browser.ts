import { Database, File, FileText, Image, Code } from '@lucide/svelte';
import type { MCPResource, MCPResourceInfo } from '$lib/types';

export interface ResourceTreeNode {
	name: string;
	resource?: MCPResourceInfo;
	children: Map<string, ResourceTreeNode>;
	isFiltered?: boolean;
}

export function parseResourcePath(uri: string): string[] {
	try {
		const withoutProtocol = uri.replace(/^[a-z]+:\/\//, '');

		return withoutProtocol.split('/').filter((p) => p.length > 0);
	} catch {
		return [uri];
	}
}

export function getDisplayName(pathPart: string): string {
	const withoutExt = pathPart.replace(/\.[^.]+$/, '');

	return withoutExt
		.split(/[-_]/)
		.map((word) => word.charAt(0).toUpperCase() + word.slice(1))
		.join(' ');
}

function resourceMatchesSearch(resource: MCPResource, query: string): boolean {
	return (
		resource.title?.toLowerCase().includes(query) || resource.uri.toLowerCase().includes(query)
	);
}

export function buildResourceTree(
	resourceList: MCPResource[],
	serverName: string,
	searchQuery?: string
): ResourceTreeNode {
	const root: ResourceTreeNode = { name: 'root', children: new Map() };

	if (!searchQuery || !searchQuery.trim()) {
		for (const resource of resourceList) {
			const pathParts = parseResourcePath(resource.uri);
			let current = root;

			for (let i = 0; i < pathParts.length - 1; i++) {
				const part = pathParts[i];
				if (!current.children.has(part)) {
					current.children.set(part, { name: part, children: new Map() });
				}
				current = current.children.get(part)!;
			}

			const fileName = pathParts[pathParts.length - 1] || resource.name;
			current.children.set(resource.uri, {
				name: fileName,
				resource: { ...resource, serverName },
				children: new Map()
			});
		}

		return root;
	}

	const query = searchQuery.toLowerCase();

	// Build tree with filtering
	for (const resource of resourceList) {
		if (!resourceMatchesSearch(resource, query)) continue;

		const pathParts = parseResourcePath(resource.uri);
		let current = root;

		for (let i = 0; i < pathParts.length - 1; i++) {
			const part = pathParts[i];
			if (!current.children.has(part)) {
				current.children.set(part, { name: part, children: new Map(), isFiltered: true });
			}
			current = current.children.get(part)!;
		}

		const fileName = pathParts[pathParts.length - 1] || resource.name;
		current.children.set(resource.uri, {
			name: fileName,
			resource: { ...resource, serverName },
			children: new Map(),
			isFiltered: true
		});
	}

	// Clean up empty folders that don't match
	function cleanupEmptyFolders(node: ResourceTreeNode): boolean {
		if (node.resource) return true;

		const toDelete: string[] = [];
		for (const [name, child] of node.children.entries()) {
			if (!cleanupEmptyFolders(child)) {
				toDelete.push(name);
			}
		}

		for (const name of toDelete) {
			node.children.delete(name);
		}

		return node.children.size > 0;
	}

	cleanupEmptyFolders(root);

	return root;
}

export function countTreeResources(node: ResourceTreeNode): number {
	if (node.resource) return 1;
	let count = 0;

	for (const child of node.children.values()) {
		count += countTreeResources(child);
	}

	return count;
}

export function getResourceIcon(resource: MCPResourceInfo) {
	const mimeType = resource.mimeType?.toLowerCase() || '';
	const uri = resource.uri.toLowerCase();

	if (mimeType.startsWith('image/') || /\.(png|jpg|jpeg|gif|svg|webp)$/.test(uri)) {
		return Image;
	}

	if (
		mimeType.includes('json') ||
		mimeType.includes('javascript') ||
		mimeType.includes('typescript') ||
		/\.(js|ts|json|yaml|yml|xml|html|css)$/.test(uri)
	) {
		return Code;
	}

	if (mimeType.includes('text') || /\.(txt|md|log)$/.test(uri)) {
		return FileText;
	}

	if (uri.includes('database') || uri.includes('db://')) {
		return Database;
	}

	return File;
}

export function sortTreeChildren(children: ResourceTreeNode[]): ResourceTreeNode[] {
	return children.sort((a, b) => {
		const aIsFolder = !a.resource && a.children.size > 0;
		const bIsFolder = !b.resource && b.children.size > 0;

		if (aIsFolder && !bIsFolder) return -1;
		if (!aIsFolder && bIsFolder) return 1;

		return a.name.localeCompare(b.name);
	});
}
