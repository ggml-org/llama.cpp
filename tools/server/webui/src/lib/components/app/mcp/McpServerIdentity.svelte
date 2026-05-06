<script lang="ts">
	import { ExternalLink } from '@lucide/svelte';
	import { Badge } from '$lib/components/ui/badge';
	import { TruncatedText } from '$lib/components/app/misc';
	import { sanitizeExternalUrl } from '$lib/utils';
	import type { MCPServerInfo } from '$lib/types';

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
		serverInfo,
		iconClass = 'h-5 w-5',
		iconRounded = 'rounded-sm',
		showVersion = true,
		showWebsite = true,
		nameClass
	}: Props = $props();

	let safeWebsiteUrl = $derived(
		serverInfo?.websiteUrl ? sanitizeExternalUrl(serverInfo.websiteUrl) : null
	);
</script>

<span class="flex min-w-0 items-center gap-1.5">
	{#if faviconUrl}
		<img
			src={faviconUrl}
			alt=""
			class={['shrink-0', iconRounded, iconClass]}
			onerror={(e) => {
				(e.currentTarget as HTMLImageElement).style.display = 'none';
			}}
		/>
	{/if}

	<TruncatedText text={displayName ?? ''} class={nameClass ?? ''} />

	{#if showVersion && serverInfo?.version}
		<Badge variant="secondary" class="h-4 min-w-0 shrink px-1 text-[10px]">
			<TruncatedText text={`v${serverInfo.version}`} />
		</Badge>
	{/if}

	{#if showWebsite && safeWebsiteUrl}
		<a
			href={safeWebsiteUrl}
			target="_blank"
			rel="noopener noreferrer"
			class="shrink-0 text-muted-foreground hover:text-foreground"
			aria-label="Open website"
			onclick={(e) => e.stopPropagation()}
		>
			<ExternalLink class="h-3 w-3" />
		</a>
	{/if}
</span>
