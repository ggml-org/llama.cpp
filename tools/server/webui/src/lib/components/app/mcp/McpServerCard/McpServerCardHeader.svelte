<script lang="ts">
	import { Cable, ExternalLink } from '@lucide/svelte';
	import { Switch } from '$lib/components/ui/switch';

	interface Props {
		displayName: string;
		faviconUrl: string | null;
		serverUrl: string;
		enabled: boolean;
		isHealthChecking: boolean;
		isConnected: boolean;
		isError: boolean;
		onToggle: (enabled: boolean) => void;
	}

	let {
		displayName,
		faviconUrl,
		serverUrl,
		enabled,
		isHealthChecking,
		isConnected,
		isError,
		onToggle
	}: Props = $props();
</script>

<div class="flex items-center justify-between gap-3">
	<div class="flex min-w-0 flex-1 items-center gap-2">
		{#if faviconUrl}
			<img
				src={faviconUrl}
				alt=""
				class="h-5 w-5 shrink-0 rounded"
				onerror={(e) => {
					(e.currentTarget as HTMLImageElement).style.display = 'none';
				}}
			/>
		{:else}
			<Cable class="h-5 w-5 shrink-0 text-muted-foreground" />
		{/if}
		<p class="truncate font-medium">{displayName}</p>
		{#if serverUrl}
			<a
				href={serverUrl}
				target="_blank"
				rel="noopener noreferrer"
				class="shrink-0 text-muted-foreground hover:text-foreground"
				aria-label="Open server URL"
			>
				<ExternalLink class="h-3.5 w-3.5" />
			</a>
		{/if}
		{#if isHealthChecking}
			<span class="shrink-0 text-xs text-muted-foreground">Checking...</span>
		{:else if isConnected}
			<span
				class="shrink-0 rounded bg-green-500/15 px-1.5 py-0.5 text-xs text-green-600 dark:text-green-500"
				>Connected</span
			>
		{:else if isError}
			<span class="shrink-0 rounded bg-destructive/15 px-1.5 py-0.5 text-xs text-destructive"
				>Error</span
			>
		{/if}
	</div>

	<div class="flex shrink-0 items-center">
		<Switch checked={enabled} onCheckedChange={onToggle} />
	</div>
</div>
