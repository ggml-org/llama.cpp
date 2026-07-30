<script lang="ts">
	import { Folder, FolderX, Loader2 } from '@lucide/svelte';
	import { AgenticSectionType } from '$lib/enums';
	import { FilesystemService } from '$lib/services';
	import { abbreviateWorkingDir, ApiError, type AgenticSection } from '$lib/utils';
	import type { ApiFilesystemRoot } from '$lib/types';
	import { parseSetWorkingDirectoryMeta } from './parsers/set-working-directory';

	interface Props {
		section: AgenticSection;
		isStreaming?: boolean;
	}

	let { section, isStreaming = false }: Props = $props();

	const isPending = $derived(section.type === AgenticSectionType.TOOL_CALL_PENDING);
	const isStreamingCall = $derived(section.type === AgenticSectionType.TOOL_CALL_STREAMING);
	const showSpinner = $derived(isPending || (isStreamingCall && isStreaming));

	const meta = $derived(parseSetWorkingDirectoryMeta(section));

	// Fetch browse roots so the working-directory display can be
	// abbreviated to `~` / `~/...` against the default root. Mirrors
	// the exec shell prefix and the picker chip - same shared utility.
	let browseRoots = $state<ApiFilesystemRoot[] | null>(null);

	async function loadBrowseRoots() {
		if (browseRoots !== null) return;
		try {
			const res = await FilesystemService.getRoots();
			browseRoots = res.roots;
		} catch (err) {
			if (!(err instanceof ApiError && err.status === 501)) {
				console.error('[SetWorkingDirectory] getRoots failed:', err);
			}
			browseRoots = [];
		}
	}

	$effect(() => {
		if (meta?.path) void loadBrowseRoots();
	});

	const setWorkingDirDisplay = $derived(abbreviateWorkingDir(meta?.path, browseRoots));
</script>

<div class="text-muted-foreground flex items-center gap-2 py-1.5">
	{#if showSpinner}
		<Folder class="text-muted-foreground/60 h-3.5 w-3.5 shrink-0" />
	{:else if meta?.errorMessage}
		<FolderX class="text-muted-foreground/60 h-3.5 w-3.5 shrink-0" />
	{:else}
		<Folder class="text-muted-foreground/60 h-3.5 w-3.5 shrink-0" />
	{/if}

	{#if showSpinner}
		<span class="text-foreground/80 text-sm font-medium">Setting working directory...</span>
		<Loader2 class="text-muted-foreground/70 h-3 w-3 animate-spin" />
	{:else if meta?.errorMessage}
		<span class="text-foreground/80 text-sm font-medium">Set working directory</span>
		<span class="text-red-600 text-xs italic dark:text-red-400">-&nbsp;{meta.errorMessage}</span>
	{:else if meta && meta.path !== null}
		<span class="text-foreground/80 text-sm font-medium">Set working directory to&nbsp;</span>
		<span class="font-mono text-foreground/90 text-sm break-all" title={meta.path}>
			{setWorkingDirDisplay}
		</span>
	{:else}
		<span class="text-foreground/80 text-sm font-medium">Working directory cleared</span>
	{/if}
</div>
