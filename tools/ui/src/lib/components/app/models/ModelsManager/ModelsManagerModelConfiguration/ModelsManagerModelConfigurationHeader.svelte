<script lang="ts">
	import { Eject, Loader2, Power, SquarePen, X } from '@lucide/svelte';
	import { ModelAvatar, ModelId } from '$lib/components/app';
	import { Button } from '$lib/components/ui/button';
	import { ServerModelStatus } from '$lib/enums';
	import type { ModelOption } from '$lib/types/models';

	interface Props {
		isLoaded: boolean;
		onClose: () => void;
		onToggleLoad: () => void;
		onUseInNewChat: () => void;
		option: ModelOption;
		/** Load state reported by the server, null when it does not report one. */
		status: ServerModelStatus | null;
	}

	let { isLoaded, onClose, onToggleLoad, onUseInNewChat, option, status }: Props = $props();

	let isLoading = $derived(status === ServerModelStatus.LOADING);

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
				<ModelId
					aliases={option.aliases}
					class="min-w-0"
					modalities={option.modalities}
					modelId={option.model}
					tags={option.tags}
					title={option.model}
				/>

				<p class="mt-1 flex items-center gap-x-2 text-xs text-muted-foreground">
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
	</div>
</header>
