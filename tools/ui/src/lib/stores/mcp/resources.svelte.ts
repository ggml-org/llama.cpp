/**
 * mcpResourceStore - Reactive State Store for MCP Resources
 *
 * Manages MCP protocol resources:
 * - Resource discovery and listing per server
 * - Resource content caching
 * - Resource subscriptions
 * - Resource attachments for chat context
 *
 * @see MCP Protocol Specification: https://modelcontextprotocol.io/specification/2025-06-18/server/resources
 */

import {
	BINARY_CONTENT_LABEL,
	MCP_RESOURCE_ATTACHMENT_ID_PREFIX,
	MCP_RESOURCE_CACHE,
	NEWLINE,
	RESOURCE_UNKNOWN_TYPE
} from '$lib/constants';
import { AttachmentType } from '$lib/enums';
import type {
	DatabaseMessageExtraMcpResource,
	MCPCachedResource,
	MCPResource,
	MCPResourceAttachment,
	MCPResourceContent,
	MCPResourceInfo,
	MCPResourceSubscription,
	MCPResourceTemplate,
	MCPResourceTemplateInfo,
	MCPServerResources
} from '$lib/types';
import { normalizeResourceUri } from '$lib/utils';
import { SvelteMap } from 'svelte/reactivity';

function generateAttachmentId(): string {
	return `${MCP_RESOURCE_ATTACHMENT_ID_PREFIX}-${Date.now()}-${Math.random().toString(36).substring(2, 9)}`;
}

class MCPResourceStore {
	private _attachments = $state<MCPResourceAttachment[]>([]);
	private _cachedResources = $state<SvelteMap<string, MCPCachedResource>>(new SvelteMap());
	private _isLoading = $state(false);
	private _serverResources = $state<SvelteMap<string, MCPServerResources>>(new SvelteMap());
	private _subscriptions = $state<SvelteMap<string, MCPResourceSubscription>>(new SvelteMap());

	get attachmentCount(): number {
		return this._attachments.length;
	}

	get attachments(): MCPResourceAttachment[] {
		return this._attachments;
	}

	get cachedResources(): Map<string, MCPCachedResource> {
		return this._cachedResources;
	}

	get hasAttachments(): boolean {
		return this._attachments.length > 0;
	}

	get isLoading(): boolean {
		return this._isLoading;
	}

	get serverResources(): Map<string, MCPServerResources> {
		return this._serverResources;
	}

	get subscriptions(): Map<string, MCPResourceSubscription> {
		return this._subscriptions;
	}

	get totalResourceCount(): number {
		let count = 0;

		for (const serverRes of this._serverResources.values()) {
			count += serverRes.resources.length;
		}

		return count;
	}

	get totalTemplateCount(): number {
		let count = 0;

		for (const serverRes of this._serverResources.values()) {
			count += serverRes.templates.length;
		}

		return count;
	}

	/**
	 * Add a resource attachment to the current chat context
	 */
	addAttachment(resource: MCPResourceInfo): MCPResourceAttachment {
		const attachment: MCPResourceAttachment = {
			id: generateAttachmentId(),
			loading: true,
			resource
		};

		this._attachments = [...this._attachments, attachment];
		console.log(`[MCPResources] Added attachment: ${resource.uri}`);

		return attachment;
	}

	/**
	 * Register a subscription for a resource
	 */
	addSubscription(uri: string, serverName: string): void {
		this._subscriptions.set(uri, {
			serverName,
			subscribedAt: new Date(),
			uri
		});

		const cached = this._cachedResources.get(uri);

		if (cached) {
			this._cachedResources.set(uri, { ...cached, subscribed: true });
		}

		console.log(`[MCPResources] Added subscription: ${uri}`);
	}

	/**
	 * Cache resource content after reading
	 */
	cacheResourceContent(resource: MCPResourceInfo, content: MCPResourceContent[]): void {
		// Enforce cache size limit
		if (this._cachedResources.size >= MCP_RESOURCE_CACHE.MAX_ENTRIES) {
			const oldestKey = this._cachedResources.keys().next().value;

			if (oldestKey) {
				this._cachedResources.delete(oldestKey);
			}
		}

		this._cachedResources.set(resource.uri, {
			content,
			fetchedAt: new Date(),
			resource,
			subscribed: this._subscriptions.has(resource.uri)
		});
		console.log(`[MCPResources] Cached content for: ${resource.uri}`);
	}

	/**
	 * Clear all state (e.g., on full reset)
	 */
	clear(): void {
		this._serverResources.clear();
		this._cachedResources.clear();
		this._subscriptions.clear();
		this._attachments = [];
		this._isLoading = false;
		console.log(`[MCPResources] Cleared all state`);
	}

	/**
	 * Clear all attachments
	 */
	clearAttachments(): void {
		this._attachments = [];
		console.log(`[MCPResources] Cleared all attachments`);
	}

	/**
	 * Clear all cached content
	 */
	clearCache(): void {
		this._cachedResources.clear();
		console.log(`[MCPResources] Cleared all cached content`);
	}

	/**
	 * Clear resources for a server (e.g., when disconnected)
	 */
	clearServerResources(serverName: string): void {
		this._serverResources.delete(serverName);

		for (const [uri, cached] of this._cachedResources) {
			if (cached.resource.serverName === serverName) {
				this._cachedResources.delete(uri);
			}
		}

		for (const [uri, sub] of this._subscriptions) {
			if (sub.serverName === serverName) {
				this._subscriptions.delete(uri);
			}
		}

		console.log(`[MCPResources][${serverName}] Cleared all resources`);
	}

	/**
	 * Find resource info by URI across all servers
	 */
	findResourceByUri(uri: string): MCPResourceInfo | undefined {
		const normalizedUri = normalizeResourceUri(uri);

		for (const [serverName, serverRes] of this._serverResources) {
			const resource =
				serverRes.resources.find((r) => r.uri === uri) ??
				serverRes.resources.find((r) => normalizeResourceUri(r.uri) === normalizedUri);

			if (resource) {
				return {
					annotations: resource.annotations,
					description: resource.description,
					icons: resource.icons,
					mimeType: resource.mimeType,
					name: resource.name,
					serverName,
					title: resource.title,
					uri: resource.uri
				};
			}
		}

		return undefined;
	}

	/**
	 * Find server name for a resource URI
	 */
	findServerForUri(uri: string): string | undefined {
		for (const [serverName, serverRes] of this._serverResources) {
			if (serverRes.resources.some((r) => r.uri === uri)) {
				return serverName;
			}
		}

		return undefined;
	}

	/**
	 * Get resource content as text for chat context
	 * Formats content for inclusion in LLM prompts
	 */
	formatAttachmentsForContext(): string {
		if (this._attachments.length === 0) return '';

		const parts: string[] = [];

		for (const attachment of this._attachments) {
			if (attachment.error) continue;

			if (!attachment.content || attachment.content.length === 0) continue;

			const resourceName = attachment.resource.title || attachment.resource.name;
			const serverName = attachment.resource.serverName;

			for (const content of attachment.content) {
				if ('text' in content && content.text) {
					parts.push(`\n\n--- Resource: ${resourceName} (from ${serverName}) ---\n${content.text}`);
				} else if ('blob' in content && content.blob) {
					// For binary content, just note it exists
					parts.push(
						`\n\n--- Resource: ${resourceName} (from ${serverName}) ---\n[${BINARY_CONTENT_LABEL}: ${content.mimeType || RESOURCE_UNKNOWN_TYPE}]`
					);
				}
			}
		}

		return parts.join('');
	}

	/**
	 * Get all resources as MCPResourceInfo array (flattened with server names)
	 */
	getAllResourceInfos(): MCPResourceInfo[] {
		const result: MCPResourceInfo[] = [];

		for (const [serverName, serverRes] of this._serverResources) {
			for (const resource of serverRes.resources) {
				result.push({
					annotations: resource.annotations,
					description: resource.description,
					icons: resource.icons,
					mimeType: resource.mimeType,
					name: resource.name,
					serverName,
					title: resource.title,
					uri: resource.uri
				});
			}
		}

		return result;
	}

	/**
	 * Get all templates as MCPResourceTemplateInfo array (flattened with server names)
	 */
	getAllTemplateInfos(): MCPResourceTemplateInfo[] {
		const result: MCPResourceTemplateInfo[] = [];

		for (const [serverName, serverRes] of this._serverResources) {
			for (const template of serverRes.templates) {
				result.push({
					annotations: template.annotations,
					description: template.description,
					icons: template.icons,
					mimeType: template.mimeType,
					name: template.name,
					serverName,
					title: template.title,
					uriTemplate: template.uriTemplate
				});
			}
		}

		return result;
	}

	/**
	 * Get attachment by ID
	 */
	getAttachment(attachmentId: string): MCPResourceAttachment | undefined {
		return this._attachments.find((att) => att.id === attachmentId);
	}

	/**
	 * Get cached content for a resource
	 */
	getCachedContent(uri: string): MCPCachedResource | undefined {
		const cached = this._cachedResources.get(uri);

		if (!cached) return undefined;

		const age = Date.now() - cached.fetchedAt.getTime();

		if (age > MCP_RESOURCE_CACHE.TTL_MS && !cached.subscribed) {
			// Cache expired and not subscribed, remove it
			this._cachedResources.delete(uri);

			return undefined;
		}

		return cached;
	}

	/**
	 * Get resources for a specific server
	 */
	getServerResources(serverName: string): MCPServerResources | undefined {
		return this._serverResources.get(serverName);
	}

	/**
	 * Handle resources list changed notification
	 */
	handleResourcesListChanged(serverName: string): void {
		const existing = this._serverResources.get(serverName);

		if (existing) {
			this._serverResources.set(serverName, {
				...existing,
				lastFetched: undefined
			});
		}

		console.log(`[MCPResources][${serverName}] Resources list changed, needs refresh`);
	}

	/**
	 *
	 *
	 * Attachments (for chat context)
	 *
	 *
	 */

	/**
	 * Handle resource update notification
	 */
	handleResourceUpdate(uri: string): void {
		// Invalidate cache so next read gets fresh content
		this.invalidateCache(uri);

		const sub = this._subscriptions.get(uri);

		if (sub) {
			this._subscriptions.set(uri, { ...sub, lastUpdate: new Date() });
		}

		console.log(`[MCPResources] Resource updated: ${uri}`);
	}

	/**
	 * Invalidate cached content for a resource (e.g., on update notification)
	 */
	invalidateCache(uri: string): void {
		this._cachedResources.delete(uri);
		console.log(`[MCPResources] Invalidated cache for: ${uri}`);
	}

	/**
	 * Check if a resource is already attached
	 */
	isAttached(uri: string): boolean {
		const normalizedUri = normalizeResourceUri(uri);

		return this._attachments.some(
			(att) => att.resource.uri === uri || normalizeResourceUri(att.resource.uri) === normalizedUri
		);
	}

	/**
	 * Check if a resource is subscribed
	 */
	isSubscribed(uri: string): boolean {
		return this._subscriptions.has(uri);
	}

	/**
	 * Remove an attachment
	 */
	removeAttachment(attachmentId: string): void {
		this._attachments = this._attachments.filter((att) => att.id !== attachmentId);
		console.log(`[MCPResources] Removed attachment: ${attachmentId}`);
	}

	/**
	 * Remove a subscription for a resource
	 */
	removeSubscription(uri: string): void {
		this._subscriptions.delete(uri);

		const cached = this._cachedResources.get(uri);

		if (cached) {
			this._cachedResources.set(uri, { ...cached, subscribed: false });
		}

		console.log(`[MCPResources] Removed subscription: ${uri}`);
	}

	/**
	 * Set global loading state
	 */
	setLoading(loading: boolean): void {
		this._isLoading = loading;
	}

	/**
	 * Set error state for a server's resources
	 */
	setServerError(serverName: string, error: string): void {
		const existing = this._serverResources.get(serverName);

		if (existing) {
			this._serverResources.set(serverName, { ...existing, error, loading: false });
		} else {
			this._serverResources.set(serverName, {
				error,
				loading: false,
				resources: [],
				serverName,
				templates: []
			});
		}
	}

	/**
	 * Set loading state for a server's resources
	 */
	setServerLoading(serverName: string, loading: boolean): void {
		const existing = this._serverResources.get(serverName);

		if (existing) {
			this._serverResources.set(serverName, { ...existing, loading });
		} else {
			this._serverResources.set(serverName, {
				error: undefined,
				loading,
				resources: [],
				serverName,
				templates: []
			});
		}
	}

	/**
	 * Set resources for a server (called after listResources)
	 */
	setServerResources(
		serverName: string,
		resources: MCPResource[],
		templates: MCPResourceTemplate[]
	): void {
		this._serverResources.set(serverName, {
			error: undefined,
			lastFetched: new Date(),
			loading: false,
			resources,
			serverName,
			templates
		});
		console.log(
			`[MCPResources][${serverName}] Set ${resources.length} resources, ${templates.length} templates`
		);
	}

	/**
	 * Convert current resource attachments to DatabaseMessageExtra[] for persisting with a message.
	 * Each attachment becomes a DatabaseMessageExtraMcpResource stored on the user message.
	 */
	toMessageExtras(): DatabaseMessageExtraMcpResource[] {
		const extras: DatabaseMessageExtraMcpResource[] = [];

		for (const attachment of this._attachments) {
			if (attachment.error) continue;

			if (!attachment.content || attachment.content.length === 0) continue;

			const resourceName = attachment.resource.title || attachment.resource.name;
			const contentParts: string[] = [];

			for (const content of attachment.content) {
				if ('text' in content && content.text) {
					contentParts.push(content.text);
				} else if ('blob' in content && content.blob) {
					contentParts.push(
						`[${BINARY_CONTENT_LABEL}: ${content.mimeType || RESOURCE_UNKNOWN_TYPE}]`
					);
				}
			}

			if (contentParts.length > 0) {
				extras.push({
					content: contentParts.join(NEWLINE),
					mimeType: attachment.resource.mimeType,
					name: resourceName,
					serverName: attachment.resource.serverName,
					type: AttachmentType.MCP_RESOURCE,
					uri: attachment.resource.uri
				});
			}
		}

		return extras;
	}

	/**
	 * Update attachment with fetched content
	 */
	updateAttachmentContent(attachmentId: string, content: MCPResourceContent[]): void {
		this._attachments = this._attachments.map((att) =>
			att.id === attachmentId ? { ...att, content, error: undefined, loading: false } : att
		);
	}

	/**
	 * Update attachment with error
	 */
	updateAttachmentError(attachmentId: string, error: string): void {
		this._attachments = this._attachments.map((att) =>
			att.id === attachmentId ? { ...att, error, loading: false } : att
		);
	}
}

export const mcpResourceStore = new MCPResourceStore();
