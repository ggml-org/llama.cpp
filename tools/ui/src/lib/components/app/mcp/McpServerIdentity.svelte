<script lang="ts">
	import { ExternalLink } from '@lucide/svelte';
	import { McpLogo } from '$lib/components/app/mcp';
	import { TruncatedText } from '$lib/components/app/misc';
	import { Badge } from '$lib/components/ui/badge';
	import type { MCPServerInfo } from '$lib/types';
	import { sanitizeExternalUrl } from '$lib/utils';

	interface Props {
		displayName?: string;
		faviconUrl?: string | null;
		serverInfo?: MCPServerInfo;
		iconClass?: string;
		iconRounded?: string;
		showVersion?: boolean;
		showWebsite?: boolean;
		nameClass?: string;
	}

	let {
		displayName,
		faviconUrl = null,
		iconClass = 'h-5 w-5',
		iconRounded = 'rounded-sm',
		nameClass,
		serverInfo,
		showVersion = true,
		showWebsite = true
	}: Props = $props();

	let safeWebsiteUrl = $derived(
		serverInfo?.websiteUrl ? sanitizeExternalUrl(serverInfo.websiteUrl) : null
	);
</script>

<span class="flex min-w-0 items-center gap-1.5">
	{#if faviconUrl}
		<img alt="" class={['shrink-0 text-foreground', iconRounded, iconClass]} src={faviconUrl} />
	{:else}
		<McpLogo class={['shrink-0 text-foreground', iconRounded, iconClass].join(' ')} />
	{/if}

	<TruncatedText class={nameClass ?? ''} text={displayName ?? ''} />

	{#if showVersion && serverInfo?.version}
		<Badge class="h-4 max-w-24 min-w-0 shrink px-1 text-[10px]" variant="secondary">
			<TruncatedText text={`v${serverInfo.version}`} />
		</Badge>
	{/if}

	{#if showWebsite && safeWebsiteUrl}
		<a
			aria-label="Open website"
			class="shrink-0 text-muted-foreground hover:text-foreground"
			href={safeWebsiteUrl}
			onclick={(e) => e.stopPropagation()}
			rel="noopener noreferrer"
			target="_blank"
		>
			<ExternalLink class="h-3 w-3" />
		</a>
	{/if}
</span>
