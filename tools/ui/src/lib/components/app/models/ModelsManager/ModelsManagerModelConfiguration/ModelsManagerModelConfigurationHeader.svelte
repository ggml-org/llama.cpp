<script lang="ts">
	import { Eject, Loader2, Power, Server, SquarePen, X } from '@lucide/svelte';
	import { Logo, ModelAvatar, ModelId } from '$lib/components/app';
	import { BackendIcon } from '$lib/components/app/backends';
	import { Badge } from '$lib/components/ui/badge';
	import { Button } from '$lib/components/ui/button';
	import { LOCAL_BACKEND_ID } from '$lib/constants';
	import { ModelCapability, ServerModelStatus } from '$lib/enums';
	import type { ModelOption } from '$lib/types/models';
	import { getBackend } from '$lib/utils/api-base';
	import { getBackendCapabilities } from '$lib/utils/backend';

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
	let backend = $derived(getBackend(option.backendId));
	let backendName = $derived(backend?.name ?? null);
	let capabilities = $derived(getBackendCapabilities(backend));
	// what the provider speaks decides which server endpoints the pane can read
	let compatTitle = $derived(
		capabilities.props
			? 'llama.cpp server: reads /props and /slots'
			: 'OpenAI-compatible: no /props, /slots, load or unload'
	);

	let isLoading = $derived(status === ServerModelStatus.LOADING);

	let statusLabel = $derived.by(() => {
		// a provider that cannot load or unload reports no load state, so its
		// models are simply on offer
		if (!capabilities.loadUnload) return 'Available';

		if (status === ServerModelStatus.LOADING) return 'Loading';

		if (status === ServerModelStatus.FAILED) return 'Failed to load';

		if (status === ServerModelStatus.SLEEPING) return 'Sleeping';

		if (isLoaded) return 'Loaded';

		return 'Not loaded';
	});

	let statusDot = $derived.by(() => {
		if (!capabilities.loadUnload) return 'bg-green-500';

		if (status === ServerModelStatus.FAILED) return 'bg-red-500';

		if (status === ServerModelStatus.LOADING) return 'bg-muted-foreground/50 animate-pulse';

		if (status === ServerModelStatus.SLEEPING) return 'bg-orange-400';

		if (isLoaded) return 'bg-green-500';

		return 'bg-muted-foreground/50';
	});
</script>

<header class="space-y-2.5 pt-3 pl-4">
	<div class="flex items-start justify-between gap-3">
		<div class="flex min-w-0 items-center gap-2">
			<!-- same geometry as the discover details header: base org, quant org badge -->
			<ModelAvatar
				{option}
				quantPositionClass="-bottom-1.5 -right-1.5"
				quantSize="h-6 w-6"
				showBaseModelAvatar
				size="h-12 w-12"
			/>

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

				<!-- one line, one voice: state, protocol, owner -->
				<p class="mt-1 flex items-center gap-x-2 text-xs text-muted-foreground">
					{#if backend && backendName}
						<span class="flex min-w-0 items-center gap-1" title={compatTitle}>
							<BackendIcon {backend} class="size-4">
								{#snippet fallback()}
									{#if backend.id === LOCAL_BACKEND_ID}
										<Logo class="shrink-0" style="--size: 0.75rem" />
									{:else}
										<Server class="size-4 shrink-0 opacity-70" />
									{/if}
								{/snippet}
							</BackendIcon>

							<!-- a long provider name shortens rather than growing the header -->
							<span class="truncate">{backendName}</span>
						</span>
					{/if}

					<span class="flex shrink-0 items-center gap-1">
						<span class="h-2 w-2 shrink-0 rounded-full {statusDot}"></span>

						{statusLabel}
					</span>
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

	<div class="flex gap-2">
		<Button class="flex-1 gap-1.5" onclick={onUseInNewChat} size="sm" variant="outline">
			<SquarePen class="h-3.5 w-3.5" />

			Start a new chat
		</Button>

		{#if capabilities.loadUnload}
			<Button
				class="flex-1 gap-1.5"
				disabled={isLoading}
				onclick={onToggleLoad}
				size="sm"
				variant="outline"
			>
				{#if isLoading}
					<Loader2 class="h-3.5 w-3.5 animate-spin" />

					Loading...
				{:else if isLoaded}
					<Eject class="h-3.5 w-3.5" />

					Unload model
				{:else}
					<Power class="h-3.5 w-3.5" />

					Load model
				{/if}
			</Button>
		{/if}
	</div>
</header>
