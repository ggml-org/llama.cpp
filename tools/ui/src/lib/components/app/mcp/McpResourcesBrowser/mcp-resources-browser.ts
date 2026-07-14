import type { MCPResource, MCPResourceInfo } from '$lib/types';
import { parseResourcePath } from '$lib/utils';

export interface ResourceTreeNode {
	name: string;
	resource?: MCPResourceInfo;
	children: Map<string, ResourceTreeNode>;
	isFiltered?: boolean;
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

export function sortTreeChildren(children: ResourceTreeNode[]): ResourceTreeNode[] {
	return children.sort((a, b) => {
		const aIsFolder = !a.resource && a.children.size > 0;
		const bIsFolder = !b.resource && b.children.size > 0;

		if (aIsFolder && !bIsFolder) return -1;
		if (!aIsFolder && bIsFolder) return 1;

		return a.name.localeCompare(b.name);
	});
}
