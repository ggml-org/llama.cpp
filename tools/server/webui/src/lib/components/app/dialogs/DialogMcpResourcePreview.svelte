<script lang="ts">
	import * as Dialog from '$lib/components/ui/dialog';
	import { Download } from '@lucide/svelte';
	import { Button } from '$lib/components/ui/button';
	import { mcpStore } from '$lib/stores/mcp.svelte';
	import { SyntaxHighlightedCode, ActionIconCopyToClipboard } from '$lib/components/app';
	import {
		getLanguageFromFilename,
		isCodeResource,
		isImageResource,
		downloadResourceContent
	} from '$lib/utils';
	import { MimeTypeIncludes, MimeTypeText } from '$lib/enums';
	import type { DatabaseMessageExtraMcpResource } from '$lib/types';

	interface Props {
		open: boolean;
		onOpenChange?: (open: boolean) => void;
		extra: DatabaseMessageExtraMcpResource;
	}

	let { open = $bindable(), onOpenChange, extra }: Props = $props();

	const serverName = $derived(mcpStore.getServerDisplayName(extra.serverName));
	const favicon = $derived(mcpStore.getServerFavicon(extra.serverName));

	function getLanguage(): string {
		if (extra.mimeType?.includes(MimeTypeIncludes.JSON)) return 'json';
		if (extra.mimeType?.includes(MimeTypeIncludes.JAVASCRIPT)) return 'javascript';
		if (extra.mimeType?.includes(MimeTypeIncludes.TYPESCRIPT)) return 'typescript';

		const name = extra.name || extra.uri || '';
		return getLanguageFromFilename(name) || 'plaintext';
	}

	function handleDownload() {
		if (!extra.content) return;
		downloadResourceContent(
			extra.content,
			extra.mimeType || MimeTypeText.PLAIN,
			extra.name || 'resource.txt'
		);
	}
</script>

<Dialog.Root bind:open {onOpenChange}>
	<Dialog.Content class="grid max-h-[90vh] max-w-5xl overflow-hidden sm:w-auto sm:max-w-6xl">
		<Dialog.Header>
			<Dialog.Title class="pr-8">{extra.name}</Dialog.Title>
			<Dialog.Description>
				<div class="flex items-center gap-2">
					<span class="text-xs text-muted-foreground">{extra.uri}</span>

					{#if serverName}
						<span class="flex items-center gap-1 text-xs text-muted-foreground">
							·
							{#if favicon}
								<img
									src={favicon}
									alt=""
									class="h-3 w-3 shrink-0 rounded-sm"
									onerror={(e) => {
										(e.currentTarget as HTMLImageElement).style.display = 'none';
									}}
								/>
							{/if}
							{serverName}
						</span>
					{/if}

					{#if extra.mimeType}
						<span class="rounded bg-muted px-1.5 py-0.5 text-xs">{extra.mimeType}</span>
					{/if}
				</div>
			</Dialog.Description>
		</Dialog.Header>

		<div class="flex items-center justify-end gap-1">
			<ActionIconCopyToClipboard
				text={extra.content}
				canCopy={!!extra.content}
				ariaLabel="Copy content"
			/>

			<Button
				variant="ghost"
				size="sm"
				class="h-7 w-7 p-0"
				onclick={handleDownload}
				disabled={!extra.content}
				title="Download content"
			>
				<Download class="h-3.5 w-3.5" />
			</Button>
		</div>

		<div class="overflow-auto">
			{#if isImageResource(extra.mimeType, extra.uri) && extra.content}
				<div class="flex items-center justify-center">
					<img
						src={extra.content.startsWith('data:')
							? extra.content
							: `data:${extra.mimeType || 'image/png'};base64,${extra.content}`}
						alt={extra.name}
						class="max-h-[70vh] max-w-full rounded object-contain"
					/>
				</div>
			{:else if isCodeResource(extra.mimeType, extra.uri) && extra.content}
				<SyntaxHighlightedCode code={extra.content} language={getLanguage()} maxHeight="70vh" />
			{:else if extra.content}
				<pre
					class="max-h-[70vh] overflow-auto rounded-md border bg-muted/30 p-4 font-mono text-sm break-words whitespace-pre-wrap">{extra.content}</pre>
			{:else}
				<div class="py-8 text-center text-sm text-muted-foreground">No content available</div>
			{/if}
		</div>
	</Dialog.Content>
</Dialog.Root>
