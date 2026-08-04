<script lang="ts">
	import { Info, Loader2 } from '@lucide/svelte';
	import { AgenticSectionType } from '$lib/enums';
	import { abbreviateHome, type AgenticSection } from '$lib/utils';
	import { toolsStore } from '$lib/stores/tools.svelte';

	interface Props {
		section: AgenticSection;
		isStreaming?: boolean;
	}

	let { section, isStreaming = false }: Props = $props();

	const isPending = $derived(section.type === AgenticSectionType.TOOL_CALL_PENDING);
	const isStreamingCall = $derived(section.type === AgenticSectionType.TOOL_CALL_STREAMING);
	const showSpinner = $derived(isPending || (isStreamingCall && isStreaming));

	type GetInfoMeta = {
		os?: string;
		cwd?: string;
		errorMessage?: string;
	};

	function parseGetInfoMeta(toolResultString: string | undefined): GetInfoMeta {
		if (!toolResultString) return {};

		try {
			const parsed: unknown = JSON.parse(toolResultString);
			if (parsed && typeof parsed === 'object' && !Array.isArray(parsed)) {
				const obj = parsed as Record<string, unknown>;
				if (typeof obj.error === 'string') return { errorMessage: obj.error };
				return {
					os: typeof obj.os === 'string' ? obj.os : undefined,
					cwd: typeof obj.cwd === 'string' ? obj.cwd : undefined
				};
			}
		} catch {
			// not JSON - nothing to show
		}

		return {};
	}

	const infoMeta = $derived(parseGetInfoMeta(section.toolResult));
	const home = $derived(toolsStore.serverHome);
	const cwdDisplay = $derived(abbreviateHome(infoMeta.cwd ?? '', home));
</script>

<div class="text-muted-foreground flex items-center gap-2 py-1.5">
	<Info class="text-muted-foreground/60 h-3.5 w-3.5 shrink-0" />
	{#if showSpinner}
		<span class="text-foreground/80 text-sm font-medium">Runtime info</span>
		<Loader2 class="text-muted-foreground/70 h-3 w-3 animate-spin" />
	{:else if infoMeta.errorMessage}
		<span class="text-foreground/80 text-sm font-medium">Runtime info&nbsp;</span>
		<span class="text-red-600 text-xs italic dark:text-red-400">-&nbsp;{infoMeta.errorMessage}</span
		>
	{:else if infoMeta.os || infoMeta.cwd}
		<span class="text-foreground/80 text-sm font-medium">Runtime info&nbsp;</span>
		{#if infoMeta.os}
			<span class="font-mono text-foreground/90 text-sm">{infoMeta.os}</span>
		{/if}
		{#if infoMeta.cwd}
			<span class="font-mono text-foreground/90 text-sm" title={infoMeta.cwd}>{cwdDisplay}</span>
		{/if}
	{:else}
		<span class="text-foreground/80 text-sm font-medium">Runtime info</span>
	{/if}
</div>
