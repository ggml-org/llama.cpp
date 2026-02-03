<script lang="ts">
	import { FileText, Loader2, AlertCircle, Download } from '@lucide/svelte';
	import { Button } from '$lib/components/ui/button';
	import { cn } from '$lib/components/ui/utils';
	import { mcpStore } from '$lib/stores/mcp.svelte';
	import { isImageMimeType, createBase64DataUrl } from '$lib/utils';
	import { MimeTypeApplication } from '$lib/enums';
	import { ActionIconCopyToClipboard } from '$lib/components/app';
	import type { MCPResourceInfo, MCPResourceContent } from '$lib/types';

	interface Props {
		resource: MCPResourceInfo | null;
		class?: string;
	}

	let { resource, class: className }: Props = $props();

	let content = $state<MCPResourceContent[] | null>(null);
	let isLoading = $state(false);
	let error = $state<string | null>(null);

	$effect(() => {
		if (resource) {
			loadContent(resource.uri);
		} else {
			content = null;
			error = null;
		}
	});

	async function loadContent(uri: string) {
		isLoading = true;
		error = null;

		try {
			const result = await mcpStore.readResource(uri);
			if (result) {
				content = result;
			} else {
				error = 'Failed to load resource content';
			}
		} catch (e) {
			error = e instanceof Error ? e.message : 'Unknown error';
		} finally {
			isLoading = false;
		}
	}

	function getTextContent(): string {
		if (!content) return '';
		return content
			.filter((c): c is { uri: string; mimeType?: string; text: string } => 'text' in c)
			.map((c) => c.text)
			.join('\n\n');
	}

	function getBlobContent(): Array<{ uri: string; mimeType?: string; blob: string }> {
		if (!content) return [];
		return content.filter(
			(c): c is { uri: string; mimeType?: string; blob: string } => 'blob' in c
		);
	}

	function handleDownload() {
		const text = getTextContent();
		if (!text || !resource) return;

		const blob = new Blob([text], { type: resource.mimeType || 'text/plain' });
		const url = URL.createObjectURL(blob);
		const a = document.createElement('a');
		a.href = url;
		a.download = resource.name || 'resource.txt';
		document.body.appendChild(a);
		a.click();
		document.body.removeChild(a);
		URL.revokeObjectURL(url);
	}
</script>

<div class={cn('flex flex-col gap-3', className)}>
	{#if !resource}
		<div class="flex flex-col items-center justify-center gap-2 py-8 text-muted-foreground">
			<FileText class="h-8 w-8 opacity-50" />

			<span class="text-sm">Select a resource to preview</span>
		</div>
	{:else}
		<div class="flex items-start justify-between gap-2">
			<div class="min-w-0 flex-1">
				<h3 class="truncate font-medium">{resource.title || resource.name}</h3>

				<p class="truncate text-xs text-muted-foreground">{resource.uri}</p>

				{#if resource.description}
					<p class="mt-1 text-sm text-muted-foreground">{resource.description}</p>
				{/if}
			</div>

			<div class="flex items-center gap-1">
				<ActionIconCopyToClipboard
					text={getTextContent()}
					canCopy={!isLoading && !!getTextContent()}
					ariaLabel="Copy content"
				/>

				<Button
					variant="ghost"
					size="sm"
					class="h-7 w-7 p-0"
					onclick={handleDownload}
					disabled={isLoading || !getTextContent()}
					title="Download content"
				>
					<Download class="h-3.5 w-3.5" />
				</Button>
			</div>
		</div>

		<div class="min-h-[200px] overflow-auto rounded-md border bg-muted/30 p-3">
			{#if isLoading}
				<div class="flex items-center justify-center py-8">
					<Loader2 class="h-6 w-6 animate-spin text-muted-foreground" />
				</div>
			{:else if error}
				<div class="flex flex-col items-center justify-center gap-2 py-8 text-red-500">
					<AlertCircle class="h-6 w-6" />

					<span class="text-sm">{error}</span>
				</div>
			{:else if content}
				{@const textContent = getTextContent()}
				{@const blobContent = getBlobContent()}

				{#if textContent}
					<pre class="font-mono text-xs break-words whitespace-pre-wrap">{textContent}</pre>
				{/if}

				{#each blobContent as blob (blob.uri)}
					{#if isImageMimeType(blob.mimeType ?? MimeTypeApplication.OCTET_STREAM)}
						<img
							src={createBase64DataUrl(
								blob.mimeType ?? MimeTypeApplication.OCTET_STREAM,
								blob.blob
							)}
							alt="Resource content"
							class="max-w-full rounded"
						/>
					{:else}
						<div class="flex items-center gap-2 rounded bg-muted p-2 text-sm text-muted-foreground">
							<FileText class="h-4 w-4" />

							<span>Binary content ({blob.mimeType || 'unknown type'})</span>
						</div>
					{/if}
				{/each}

				{#if !textContent && blobContent.length === 0}
					<div class="py-4 text-center text-sm text-muted-foreground">No content available</div>
				{/if}
			{/if}
		</div>

		{#if resource.mimeType || resource.annotations}
			<div class="flex flex-wrap gap-2 text-xs text-muted-foreground">
				{#if resource.mimeType}
					<span class="rounded bg-muted px-1.5 py-0.5">{resource.mimeType}</span>
				{/if}

				{#if resource.annotations?.priority !== undefined}
					<span class="rounded bg-muted px-1.5 py-0.5">
						Priority: {resource.annotations.priority}
					</span>
				{/if}

				<span class="rounded bg-muted px-1.5 py-0.5">
					Server: {resource.serverName}
				</span>
			</div>
		{/if}
	{/if}
</div>
