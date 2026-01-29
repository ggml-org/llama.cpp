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

import { SvelteMap } from 'svelte/reactivity';
import type {
	MCPResource,
	MCPResourceTemplate,
	MCPResourceContent,
	MCPResourceInfo,
	MCPResourceTemplateInfo,
	MCPCachedResource,
	MCPResourceAttachment,
	MCPResourceSubscription,
	MCPServerResources
} from '$lib/types';

const MAX_CACHED_RESOURCES = 50;
const CACHE_TTL_MS = 5 * 60 * 1000; // 5 minutes

function generateAttachmentId(): string {
	return `res-${Date.now()}-${Math.random().toString(36).substring(2, 9)}`;
}

class MCPResourceStore {
	private _serverResources = $state<SvelteMap<string, MCPServerResources>>(new SvelteMap());
	private _cachedResources = $state<SvelteMap<string, MCPCachedResource>>(new SvelteMap());
	private _subscriptions = $state<SvelteMap<string, MCPResourceSubscription>>(new SvelteMap());
	private _attachments = $state<MCPResourceAttachment[]>([]);
	private _isLoading = $state(false);

	get serverResources(): Map<string, MCPServerResources> {
		return this._serverResources;
	}

	get cachedResources(): Map<string, MCPCachedResource> {
		return this._cachedResources;
	}

	get subscriptions(): Map<string, MCPResourceSubscription> {
		return this._subscriptions;
	}

	get attachments(): MCPResourceAttachment[] {
		return this._attachments;
	}

	get isLoading(): boolean {
		return this._isLoading;
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

	get attachmentCount(): number {
		return this._attachments.length;
	}

	get hasAttachments(): boolean {
		return this._attachments.length > 0;
	}

	/**
	 *
	 *
	 * Server Resources Management
	 *
	 *
	 */

	/**
	 * Set resources for a server (called after listResources)
	 */
	setServerResources(
		serverName: string,
		resources: MCPResource[],
		templates: MCPResourceTemplate[]
	): void {
		this._serverResources.set(serverName, {
			serverName,
			resources,
			templates,
			lastFetched: new Date(),
			loading: false,
			error: undefined
		});
		console.log(
			`[MCPResources][${serverName}] Set ${resources.length} resources, ${templates.length} templates`
		);
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
				serverName,
				resources: [],
				templates: [],
				loading,
				error: undefined
			});
		}
	}

	/**
	 * Set error state for a server's resources
	 */
	setServerError(serverName: string, error: string): void {
		const existing = this._serverResources.get(serverName);
		if (existing) {
			this._serverResources.set(serverName, { ...existing, loading: false, error });
		} else {
			this._serverResources.set(serverName, {
				serverName,
				resources: [],
				templates: [],
				loading: false,
				error
			});
		}
	}

	/**
	 * Get resources for a specific server
	 */
	getServerResources(serverName: string): MCPServerResources | undefined {
		return this._serverResources.get(serverName);
	}

	/**
	 * Get all resources as MCPResourceInfo array (flattened with server names)
	 */
	getAllResourceInfos(): MCPResourceInfo[] {
		const result: MCPResourceInfo[] = [];
		for (const [serverName, serverRes] of this._serverResources) {
			for (const resource of serverRes.resources) {
				result.push({
					uri: resource.uri,
					name: resource.name,
					title: resource.title,
					description: resource.description,
					mimeType: resource.mimeType,
					serverName,
					annotations: resource.annotations,
					icons: resource.icons
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
					uriTemplate: template.uriTemplate,
					name: template.name,
					title: template.title,
					description: template.description,
					mimeType: template.mimeType,
					serverName,
					annotations: template.annotations,
					icons: template.icons
				});
			}
		}
		return result;
	}

	/**
	 * Clear resources for a server (e.g., when disconnected)
	 */
	clearServerResources(serverName: string): void {
		this._serverResources.delete(serverName);
		// Also clear cached content for this server's resources
		for (const [uri, cached] of this._cachedResources) {
			if (cached.resource.serverName === serverName) {
				this._cachedResources.delete(uri);
			}
		}
		// Clear subscriptions for this server
		for (const [uri, sub] of this._subscriptions) {
			if (sub.serverName === serverName) {
				this._subscriptions.delete(uri);
			}
		}
		console.log(`[MCPResources][${serverName}] Cleared all resources`);
	}

	/**
	 *
	 *
	 * Resource Content Caching
	 *
	 *
	 */

	/**
	 * Cache resource content after reading
	 */
	cacheResourceContent(resource: MCPResourceInfo, content: MCPResourceContent[]): void {
		// Enforce cache size limit
		if (this._cachedResources.size >= MAX_CACHED_RESOURCES) {
			// Remove oldest entry
			const oldestKey = this._cachedResources.keys().next().value;
			if (oldestKey) {
				this._cachedResources.delete(oldestKey);
			}
		}

		this._cachedResources.set(resource.uri, {
			resource,
			content,
			fetchedAt: new Date(),
			subscribed: this._subscriptions.has(resource.uri)
		});
		console.log(`[MCPResources] Cached content for: ${resource.uri}`);
	}

	/**
	 * Get cached content for a resource
	 */
	getCachedContent(uri: string): MCPCachedResource | undefined {
		const cached = this._cachedResources.get(uri);
		if (!cached) return undefined;

		// Check if cache is still valid
		const age = Date.now() - cached.fetchedAt.getTime();
		if (age > CACHE_TTL_MS && !cached.subscribed) {
			// Cache expired and not subscribed, remove it
			this._cachedResources.delete(uri);
			return undefined;
		}

		return cached;
	}

	/**
	 * Invalidate cached content for a resource (e.g., on update notification)
	 */
	invalidateCache(uri: string): void {
		this._cachedResources.delete(uri);
		console.log(`[MCPResources] Invalidated cache for: ${uri}`);
	}

	/**
	 * Clear all cached content
	 */
	clearCache(): void {
		this._cachedResources.clear();
		console.log(`[MCPResources] Cleared all cached content`);
	}

	/**
	 *
	 *
	 * Subscriptions
	 *
	 *
	 */

	/**
	 * Register a subscription for a resource
	 */
	addSubscription(uri: string, serverName: string): void {
		this._subscriptions.set(uri, {
			uri,
			serverName,
			subscribedAt: new Date()
		});

		// Update cached resource if exists
		const cached = this._cachedResources.get(uri);
		if (cached) {
			this._cachedResources.set(uri, { ...cached, subscribed: true });
		}

		console.log(`[MCPResources] Added subscription: ${uri}`);
	}

	/**
	 * Remove a subscription for a resource
	 */
	removeSubscription(uri: string): void {
		this._subscriptions.delete(uri);

		// Update cached resource if exists
		const cached = this._cachedResources.get(uri);
		if (cached) {
			this._cachedResources.set(uri, { ...cached, subscribed: false });
		}

		console.log(`[MCPResources] Removed subscription: ${uri}`);
	}

	/**
	 * Check if a resource is subscribed
	 */
	isSubscribed(uri: string): boolean {
		return this._subscriptions.has(uri);
	}

	/**
	 * Handle resource update notification
	 */
	handleResourceUpdate(uri: string): void {
		// Invalidate cache so next read gets fresh content
		this.invalidateCache(uri);

		// Update subscription last update time
		const sub = this._subscriptions.get(uri);
		if (sub) {
			this._subscriptions.set(uri, { ...sub, lastUpdate: new Date() });
		}

		console.log(`[MCPResources] Resource updated: ${uri}`);
	}

	/**
	 * Handle resources list changed notification
	 */
	handleResourcesListChanged(serverName: string): void {
		// Mark server resources as needing refresh
		const existing = this._serverResources.get(serverName);
		if (existing) {
			this._serverResources.set(serverName, {
				...existing,
				lastFetched: undefined // Mark as stale
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
	 * Add a resource attachment to the current chat context
	 */
	addAttachment(resource: MCPResourceInfo): MCPResourceAttachment {
		const attachment: MCPResourceAttachment = {
			id: generateAttachmentId(),
			resource,
			loading: true
		};

		this._attachments = [...this._attachments, attachment];
		console.log(`[MCPResources] Added attachment: ${resource.uri}`);

		return attachment;
	}

	/**
	 * Update attachment with fetched content
	 */
	updateAttachmentContent(attachmentId: string, content: MCPResourceContent[]): void {
		this._attachments = this._attachments.map((att) =>
			att.id === attachmentId ? { ...att, content, loading: false, error: undefined } : att
		);
	}

	/**
	 * Update attachment with error
	 */
	updateAttachmentError(attachmentId: string, error: string): void {
		this._attachments = this._attachments.map((att) =>
			att.id === attachmentId ? { ...att, loading: false, error } : att
		);
	}

	/**
	 * Remove an attachment
	 */
	removeAttachment(attachmentId: string): void {
		this._attachments = this._attachments.filter((att) => att.id !== attachmentId);
		console.log(`[MCPResources] Removed attachment: ${attachmentId}`);
	}

	/**
	 * Clear all attachments
	 */
	clearAttachments(): void {
		this._attachments = [];
		console.log(`[MCPResources] Cleared all attachments`);
	}

	/**
	 * Get attachment by ID
	 */
	getAttachment(attachmentId: string): MCPResourceAttachment | undefined {
		return this._attachments.find((att) => att.id === attachmentId);
	}

	/**
	 * Check if a resource is already attached
	 */
	isAttached(uri: string): boolean {
		return this._attachments.some((att) => att.resource.uri === uri);
	}

	/**
	 *
	 *
	 * Utility Methods
	 *
	 *
	 */

	/**
	 * Set global loading state
	 */
	setLoading(loading: boolean): void {
		this._isLoading = loading;
	}

	/**
	 * Find resource info by URI across all servers
	 */
	findResourceByUri(uri: string): MCPResourceInfo | undefined {
		for (const [serverName, serverRes] of this._serverResources) {
			const resource = serverRes.resources.find((r) => r.uri === uri);
			if (resource) {
				return {
					uri: resource.uri,
					name: resource.name,
					title: resource.title,
					description: resource.description,
					mimeType: resource.mimeType,
					serverName,
					annotations: resource.annotations,
					icons: resource.icons
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
						`\n\n--- Resource: ${resourceName} (from ${serverName}) ---\n[Binary content: ${content.mimeType || 'unknown type'}]`
					);
				}
			}
		}

		return parts.join('');
	}
}

export const mcpResourceStore = new MCPResourceStore();

// Export convenience functions
export const mcpResources = () => mcpResourceStore.serverResources;
export const mcpResourceAttachments = () => mcpResourceStore.attachments;
export const mcpResourceAttachmentCount = () => mcpResourceStore.attachmentCount;
export const mcpHasResourceAttachments = () => mcpResourceStore.hasAttachments;
export const mcpTotalResourceCount = () => mcpResourceStore.totalResourceCount;
export const mcpResourcesLoading = () => mcpResourceStore.isLoading;
