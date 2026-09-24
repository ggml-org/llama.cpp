<script lang="ts">
	import { resolveModelSize } from '../utils';
	import { Eject, HardDrive, Power, ScrollText, Server, SquarePen, X } from '@lucide/svelte';
	import { ModelAvatar, ModelContext, ModelId } from '$lib/components/app';
	import { Badge } from '$lib/components/ui/badge';
	import { Button } from '$lib/components/ui/button';
	import { ModelCapability, ServerModelStatus } from '$lib/enums';
	import type { ModelOption } from '$lib/types/models';
	import { getBackend } from '$lib/utils/api-base';

	interface Props {
		isCustomized: boolean;
		isLoaded: boolean;
		onClose: () => void;
		onToggleLoad: () => void;
		onUseInNewChat: () => void;
		option: ModelOption;
		/** Load state reported by the server, null when it does not report one. */
		status: ServerModelStatus | null;
	}

	let { isCustomized, isLoaded, onClose, onToggleLoad, onUseInNewChat, option, status }: Props =
		$props();

	let supportsToolUse = $derived(option.capabilities.includes(ModelCapability.TOOL_USE));
	let supportsThinking = $derived(option.capabilities.includes(ModelCapability.REASONING));
	let backendName = $derived(getBackend(option.backendId)?.name ?? null);

	// the listing usually carries the size; a local repo falls back to its tree
	let size = $state<string | null>(null);

	$effect(() => {
		let cancelled = false;

		size = null;

		void resolveModelSize(option)
			.then((label) => {
				if (!cancelled) size = label;
			})
			.catch(() => {});

		return () => {
			cancelled = true;
		};
	});

	let statusLabel = $derived.by(() => {
		if (status === ServerModelStatus.LOADING) return 'Loading';

		if (status === ServerModelStatus.FAILED) return 'Failed to load';

		if (status === ServerModelStatus.SLEEPING) return 'Sleeping';

		if (isLoaded) return 'Loaded';

		return 'Not loaded';
	});

	let statusDot = $derived.by(() => {
		if (status === ServerModelStatus.FAILED) return 'bg-red-500';

		if (status === ServerModelStatus.LOADING) return 'bg-muted-foreground/50 animate-pulse';

		if (status === ServerModelStatus.SLEEPING) return 'bg-orange-400';

		if (isLoaded) return 'bg-green-500';

		return 'bg-muted-foreground/50';
	});
</script>

<header class="space-y-3 px-4 pt-4">
	<div class="flex items-start justify-between gap-3">
		<div class="flex min-w-0 items-center gap-2">
			<ModelAvatar {option} showBaseModelAvatar size="size-12" />

			<div class="min-w-0">
				<div class="flex items-center gap-2">
					<ModelId
						aliases={option.aliases}
						class="min-w-0"
						modalities={option.modalities}
						modelId={option.model}
						{supportsThinking}
						{supportsToolUse}
						tags={option.tags}
						title={option.model}
					/>

					{#if isCustomized}
						<Badge class="h-5 shrink-0 px-1.5 text-[10px]" variant="secondary">CUSTOMIZED</Badge>
					{/if}
				</div>

				<p class="mt-0.5 flex items-center gap-1.5 text-xs text-muted-foreground">
					<span class="h-2 w-2 shrink-0 rounded-full {statusDot}"></span>

					{statusLabel}
				</p>
			</div>
		</div>

		<Button
			aria-label="Close details"
			class="h-7 w-7"
			onclick={onClose}
			size="icon"
			variant="ghost"
		>
			<X class="h-4 w-4" />
		</Button>
	</div>

	<div class="flex flex-wrap items-center gap-x-3 gap-y-1 text-xs text-muted-foreground">
		<span class="flex items-center gap-1.5" title="Context window">
			<ScrollText class="h-3.5 w-3.5 shrink-0" />

			<ModelContext {option} />
		</span>

		{#if size}
			<span class="flex items-center gap-1.5" title="Size on disk">
				<HardDrive class="h-3.5 w-3.5 shrink-0" />

				{size}
			</span>
		{/if}

		{#if backendName}
			<span class="flex items-center gap-1.5" title="Served by">
				<Server class="h-3.5 w-3.5 shrink-0" />

				{backendName}
			</span>
		{/if}
	</div>

	<div class="flex gap-2">
		<Button class="flex-1 gap-1.5" onclick={onUseInNewChat} variant="outline">
			<SquarePen class="h-3.5 w-3.5" />

			Start a new chat
		</Button>

		<Button class="flex-1 gap-1.5" onclick={onToggleLoad} variant="outline">
			{#if isLoaded}
				<Eject class="h-3.5 w-3.5" />

				Unload model
			{:else}
				<Power class="h-3.5 w-3.5" />

				Load model
			{/if}
		</Button>
	</div>
</header>
