<script lang="ts">
	import ModelsManagerModelConfigurationActions from './ModelsManagerModelConfigurationActions.svelte';
	import { X } from '@lucide/svelte';
	import { ModelAvatar, ModelId } from '$lib/components/app';
	import { Button } from '$lib/components/ui/button';
	import { ServerModelStatus } from '$lib/enums';
	import { deviceStore } from '$lib/stores';
	import type { ModelOption } from '$lib/types/models';

	interface Props {
		isLoaded: boolean;
		onClose: () => void;
		onToggleLoad: () => void;
		onUseInChat: () => void;
		onUseInNewChat: () => void;
		option: ModelOption;
		/** Load state reported by the server, null when it does not report one. */
		status: ServerModelStatus | null;
	}

	let { isLoaded, onClose, onToggleLoad, onUseInChat, onUseInNewChat, option, status }: Props =
		$props();

	let isMobile = $derived(deviceStore.isMobile);
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

<!-- only the desktop pane leaves the right edge to the panel border: on a phone the
     pane covers the screen, so its two sides carry the same padding, and it takes a
     little more room around the model it names -->
<header class="space-y-2.5 pt-3 pl-4 max-md:space-y-4 max-md:pt-4 max-md:pr-4 max-md:pb-4">
	<div class="flex items-start justify-between gap-3">
		<div class="flex min-w-0 items-center gap-3">
			<!-- same geometry as the discover details header: base org, quant org badge -->
			<ModelAvatar
				{option}
				quantPositionClass="-bottom-1.5 -right-1.5"
				quantSize="h-6 w-6"
				showBaseModelAvatar
				size="h-12 w-12"
			/>

			<div class="min-w-0">
				<!-- a phone gives the id a row of its own, with its badges and icons under it -->
				<ModelId
					aliases={option.aliases}
					class="min-w-0"
					modalities={option.modalities}
					modelId={option.model}
					stackId={deviceStore.isMobile}
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
			class="h-7 w-7 max-md:-mt-3 max-md:-mr-3 max-md:h-10 max-md:w-10 max-md:rounded-full"
			onclick={onClose}
			size="icon"
			variant="ghost"
		>
			<X class="h-4 w-4" />
		</Button>
	</div>

	<!-- a phone floats the actions at the bottom of the pane, where the thumb is -->
	{#if !isMobile}
		<ModelsManagerModelConfigurationActions
			{isLoaded}
			{isLoading}
			{onToggleLoad}
			{onUseInChat}
			{onUseInNewChat}
			size="default"
		/>
	{/if}
</header>
