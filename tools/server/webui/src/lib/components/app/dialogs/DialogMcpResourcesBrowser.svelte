<script lang="ts">
	import { FolderOpen, Plus, Loader2, Braces } from '@lucide/svelte';
	import { toast } from 'svelte-sonner';
	import * as Dialog from '$lib/components/ui/dialog';
	import { Button } from '$lib/components/ui/button';
	import { mcpStore } from '$lib/stores/mcp.svelte';
	import { conversationsStore } from '$lib/stores/conversations.svelte';
	import {
		mcpResources,
		mcpTotalResourceCount,
		mcpResourceStore
	} from '$lib/stores/mcp-resources.svelte';
	import {
		McpResourcesBrowser,
		McpResourcePreview,
		McpResourceTemplateForm
	} from '$lib/components/app';
	import { getResourceDisplayName } from '$lib/utils';
	import type { MCPResourceInfo, MCPResourceContent, MCPResourceTemplateInfo } from '$lib/types';
	import { SvelteSet } from 'svelte/reactivity';

	interface Props {
		open?: boolean;
		onOpenChange?: (open: boolean) => void;
		onAttach?: (resource: MCPResourceInfo) => void;
		preSelectedUri?: string;
	}

	let { open = $bindable(false), onOpenChange, onAttach, preSelectedUri }: Props = $props();

	let selectedResources = new SvelteSet<string>();
	let lastSelectedUri = $state<string | null>(null);
	let isAttaching = $state(false);

	let selectedTemplate = $state<MCPResourceTemplateInfo | null>(null);
	let templatePreviewUri = $state<string | null>(null);
	let templatePreviewContent = $state<MCPResourceContent[] | null>(null);
	let templatePreviewLoading = $state(false);
	let templatePreviewError = $state<string | null>(null);

	const totalCount = $derived(mcpTotalResourceCount());

	$effect(() => {
		if (open) {
			loadResources();

			if (preSelectedUri) {
				selectedResources.clear();
				selectedResources.add(preSelectedUri);
				lastSelectedUri = preSelectedUri;
			}
		}
	});

	async function loadResources() {
		const perChatOverrides = conversationsStore.getAllMcpServerOverrides();
		const initialized = await mcpStore.ensureInitialized(perChatOverrides);

		if (initialized) {
			await mcpStore.fetchAllResources();
		}
	}

	function handleOpenChange(newOpen: boolean) {
		open = newOpen;
		onOpenChange?.(newOpen);

		if (!newOpen) {
			selectedResources.clear();
			lastSelectedUri = null;
			clearTemplateState();
		}
	}

	function clearTemplateState() {
		selectedTemplate = null;
		templatePreviewUri = null;
		templatePreviewContent = null;
		templatePreviewLoading = false;
		templatePreviewError = null;
	}

	function handleTemplateSelect(template: MCPResourceTemplateInfo) {
		selectedResources.clear();
		lastSelectedUri = null;

		if (
			selectedTemplate?.uriTemplate === template.uriTemplate &&
			selectedTemplate?.serverName === template.serverName
		) {
			clearTemplateState();

			return;
		}

		selectedTemplate = template;
		templatePreviewUri = null;
		templatePreviewContent = null;
		templatePreviewLoading = false;
		templatePreviewError = null;
	}

	async function handleTemplateResolve(uri: string, serverName: string) {
		templatePreviewUri = uri;
		templatePreviewContent = null;
		templatePreviewLoading = true;
		templatePreviewError = null;

		try {
			const content = await mcpStore.readResourceByUri(serverName, uri);

			if (content) {
				templatePreviewContent = content;
			} else {
				templatePreviewError = 'Failed to read resource';
			}
		} catch (error) {
			templatePreviewError = error instanceof Error ? error.message : 'Unknown error';
		} finally {
			templatePreviewLoading = false;
		}
	}

	function handleTemplateCancelForm() {
		clearTemplateState();
	}

	async function handleAttachTemplateResource() {
		if (!templatePreviewUri || !selectedTemplate || !templatePreviewContent) return;

		isAttaching = true;

		try {
			const knownResource = mcpResourceStore.findResourceByUri(templatePreviewUri);

			if (knownResource) {
				if (!mcpResourceStore.isAttached(knownResource.uri)) {
					await mcpStore.attachResource(knownResource.uri);
				}

				toast.success(`Resource attached: ${knownResource.title || knownResource.name}`);
			} else {
				if (mcpResourceStore.isAttached(templatePreviewUri)) {
					toast.info('Resource already attached');
					handleOpenChange(false);
					return;
				}

				const resourceInfo: MCPResourceInfo = {
					uri: templatePreviewUri,
					name: templatePreviewUri.split('/').pop() || templatePreviewUri,
					serverName: selectedTemplate.serverName
				};

				const attachment = mcpResourceStore.addAttachment(resourceInfo);
				mcpResourceStore.updateAttachmentContent(attachment.id, templatePreviewContent);

				toast.success(`Resource attached: ${resourceInfo.name}`);
			}

			handleOpenChange(false);
		} catch (error) {
			console.error('Failed to attach template resource:', error);
		} finally {
			isAttaching = false;
		}
	}

	function handleResourceSelect(resource: MCPResourceInfo, shiftKey: boolean = false) {
		clearTemplateState();

		if (shiftKey && lastSelectedUri) {
			const allResources = getAllResourcesFlatInTreeOrder();
			const lastIndex = allResources.findIndex((r) => r.uri === lastSelectedUri);
			const currentIndex = allResources.findIndex((r) => r.uri === resource.uri);

			if (lastIndex !== -1 && currentIndex !== -1) {
				const start = Math.min(lastIndex, currentIndex);
				const end = Math.max(lastIndex, currentIndex);

				for (let i = start; i <= end; i++) {
					selectedResources.add(allResources[i].uri);
				}
			}
		} else {
			selectedResources.clear();
			selectedResources.add(resource.uri);
			lastSelectedUri = resource.uri;
		}
	}

	function handleResourceToggle(resource: MCPResourceInfo, checked: boolean) {
		clearTemplateState();

		if (checked) {
			selectedResources.add(resource.uri);
		} else {
			selectedResources.delete(resource.uri);
		}

		lastSelectedUri = resource.uri;
	}

	function getAllResourcesFlatInTreeOrder(): MCPResourceInfo[] {
		const allResources: MCPResourceInfo[] = [];
		const resourcesMap = mcpResources();

		for (const [serverName, serverRes] of resourcesMap.entries()) {
			for (const resource of serverRes.resources) {
				allResources.push({ ...resource, serverName });
			}
		}

		return allResources.sort((a, b) => {
			const aName = getResourceDisplayName(a);
			const bName = getResourceDisplayName(b);
			return aName.localeCompare(bName);
		});
	}

	async function handleAttach() {
		if (selectedResources.size === 0) return;

		isAttaching = true;

		try {
			const allResources = getAllResourcesFlatInTreeOrder();
			const resourcesToAttach = allResources.filter((r) => selectedResources.has(r.uri));

			for (const resource of resourcesToAttach) {
				await mcpStore.attachResource(resource.uri);
				onAttach?.(resource);
			}

			const count = resourcesToAttach.length;

			toast.success(
				count === 1
					? `Resource attached: ${resourcesToAttach[0].name}`
					: `${count} resources attached`
			);

			handleOpenChange(false);
		} catch (error) {
			console.error('Failed to attach resources:', error);
		} finally {
			isAttaching = false;
		}
	}

	const selectedTemplateUri = $derived(selectedTemplate?.uriTemplate ?? null);

	const hasTemplateResult = $derived(
		!!selectedTemplate && !!templatePreviewContent && !!templatePreviewUri
	);
</script>

<Dialog.Root {open} onOpenChange={handleOpenChange}>
	<Dialog.Content class="max-h-[80vh] !max-w-4xl overflow-hidden p-0">
		<Dialog.Header class="border-b border-border/30 px-6 py-4">
			<Dialog.Title class="flex items-center gap-2">
				<FolderOpen class="h-5 w-5" />

				<span>MCP Resources</span>

				{#if totalCount > 0}
					<span class="text-sm font-normal text-muted-foreground">({totalCount})</span>
				{/if}
			</Dialog.Title>

			<Dialog.Description>
				Browse and attach resources from connected MCP servers to your chat context.
			</Dialog.Description>
		</Dialog.Header>

		<div class="flex h-[500px] min-w-0">
			<div class="w-72 shrink-0 overflow-y-auto border-r border-border/30 p-4">
				<McpResourcesBrowser
					onSelect={handleResourceSelect}
					onToggle={handleResourceToggle}
					onTemplateSelect={handleTemplateSelect}
					selectedUris={selectedResources}
					{selectedTemplateUri}
					expandToUri={preSelectedUri}
				/>
			</div>

			<div class="min-w-0 flex-1 overflow-auto p-4">
				{#if selectedTemplate && !templatePreviewContent}
					<div class="flex h-full flex-col">
						<div class="mb-3 flex items-center gap-2">
							<Braces class="h-4 w-4 text-muted-foreground" />

							<span class="text-sm font-medium">
								{selectedTemplate.title || selectedTemplate.name}
							</span>
						</div>

						{#if selectedTemplate.description}
							<p class="mb-4 text-xs text-muted-foreground">
								{selectedTemplate.description}
							</p>
						{/if}

						<div class="mb-4 rounded-md border border-border/50 bg-muted/30 px-3 py-2">
							<p class="font-mono text-xs break-all text-muted-foreground">
								{selectedTemplate.uriTemplate}
							</p>
						</div>

						{#if templatePreviewLoading}
							<div class="flex flex-1 items-center justify-center">
								<Loader2 class="h-6 w-6 animate-spin text-muted-foreground" />
							</div>
						{:else if templatePreviewError}
							<div class="flex flex-1 flex-col items-center justify-center gap-2 text-red-500">
								<span class="text-sm">{templatePreviewError}</span>

								<Button
									size="sm"
									variant="outline"
									onclick={() => {
										templatePreviewError = null;
									}}
								>
									Try again
								</Button>
							</div>
						{:else}
							<McpResourceTemplateForm
								template={selectedTemplate}
								onResolve={handleTemplateResolve}
								onCancel={handleTemplateCancelForm}
							/>
						{/if}
					</div>
				{:else if hasTemplateResult}
					<!-- Template resolved: show preview -->
					<McpResourcePreview
						resource={{
							uri: templatePreviewUri ?? '',
							name: templatePreviewUri?.split('/').pop() || (templatePreviewUri ?? ''),
							serverName: selectedTemplate?.serverName || ''
						}}
						preloadedContent={templatePreviewContent}
					/>
				{:else if selectedResources.size === 1}
					{@const allResources = getAllResourcesFlatInTreeOrder()}
					{@const selectedResource = allResources.find((r) => selectedResources.has(r.uri))}

					<McpResourcePreview resource={selectedResource ?? null} />
				{:else if selectedResources.size > 1}
					<div class="flex flex-col gap-10">
						{#each getAllResourcesFlatInTreeOrder() as resource (resource.uri)}
							{#if selectedResources.has(resource.uri)}
								<McpResourcePreview {resource} />
							{/if}
						{/each}
					</div>
				{:else}
					<div class="flex h-full items-center justify-center text-sm text-muted-foreground">
						Select a resource to preview
					</div>
				{/if}
			</div>
		</div>

		<Dialog.Footer class="border-t border-border/30 px-6 py-4">
			<Button variant="outline" onclick={() => handleOpenChange(false)}>Cancel</Button>

			{#if hasTemplateResult}
				<Button onclick={handleAttachTemplateResource} disabled={isAttaching}>
					{#if isAttaching}
						<Loader2 class="mr-2 h-4 w-4 animate-spin" />
					{:else}
						<Plus class="mr-2 h-4 w-4" />
					{/if}

					Attach Resource
				</Button>
			{:else}
				<Button onclick={handleAttach} disabled={selectedResources.size === 0 || isAttaching}>
					{#if isAttaching}
						<Loader2 class="mr-2 h-4 w-4 animate-spin" />
					{:else}
						<Plus class="mr-2 h-4 w-4" />
					{/if}

					Attach {selectedResources.size > 0 ? `(${selectedResources.size})` : 'Resource'}
				</Button>
			{/if}
		</Dialog.Footer>
	</Dialog.Content>
</Dialog.Root>
