<script lang="ts">
	import ModelsManagerModelConfigurationActions from './ModelsManagerModelConfigurationActions.svelte';
	import { X } from '@lucide/svelte';
	import { ModelAvatar, ModelId } from '$lib/components/app';
	import { Button } from '$lib/components/ui/button';
	import { ServerModelStatus } from '$lib/enums';
	import { deviceStore } from '$lib/stores';
	import type { ModelOption } from '$lib/types/models';

	interface Props {
		isDownload?: boolean;
		isDownloadPaused?: boolean;
		isLoaded: boolean;
		onClose: () => void;
		/** Opens the manager dialog on the Discover view, focused on this model's repo. */
		onSeeInDiscover?: () => void;
		onRequestCancelDownload?: () => void;
		onToggleDownload?: () => void;
		onToggleLoad: () => void;
		onUseInChat: () => void;
		onUseInNewChat: () => void;
		option: ModelOption;
		/** Load state reported by the server, null when it does not report one. */
		status: ServerModelStatus | null;
	}

	let {
		isDownload = false,
		isDownloadPaused = false,
		isLoaded,
		onClose,
		onRequestCancelDownload,
		onSeeInDiscover,
		onToggleDownload,
		onToggleLoad,
		onUseInChat,
		onUseInNewChat,
		option,
		status
	}: Props = $props();

	let isMobile = $derived(deviceStore.isMobile);
	let isLoading = $derived(status === ServerModelStatus.LOADING);

	let statusLabel = $derived.by(() => {
		if (isDownload) return isDownloadPaused ? 'Paused' : 'Downloading';

		if (status === ServerModelStatus.LOADING) return 'Loading';

		if (status === ServerModelStatus.FAILED) return 'Failed to load';

		if (status === ServerModelStatus.SLEEPING) return 'Sleeping';

		if (isLoaded) return 'Loaded';

		return 'Not loaded';
	});

	let statusDot = $derived.by(() => {
		if (isDownload)
			return isDownloadPaused ? 'status-lamp orb-warning' : 'status-lamp orb-blue animate-pulse';

		if (status === ServerModelStatus.FAILED) return 'status-lamp orb-danger';

		if (status === ServerModelStatus.LOADING) return 'bg-muted-foreground/50 animate-pulse';

		if (status === ServerModelStatus.SLEEPING) return 'status-lamp orb-warning';

		if (isLoaded) return 'status-lamp orb-success';

		return 'bg-muted-foreground/50';
	});
</script>

<!-- only the desktop pane leaves the right edge to the panel border: on a phone the
     pane covers the screen, so its two sides carry the same padding, and it takes a
     little more room around the model it names -->
<header class="space-y-2.5 px-4 pt-3 max-md:space-y-4 max-md:pt-4 max-md:pb-4">
	<div class="flex items-start justify-between gap-3">
		<div class="flex min-w-0 items-center gap-3">
			<!-- same geometry as the discover details header: base org, quant org badge -->
			<ModelAvatar
				{option}
				quantPositionClass="-bottom-1.5 -right-1.5"
				quantSize="size-6"
				showBaseModelAvatar
				size="size-12"
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
						<span class="size-2 shrink-0 rounded-full {statusDot}"></span>

						{statusLabel}
					</span>
				</p>
			</div>
		</div>

		<!-- a phone's bottom drawer closes by its handle and overlay, it needs no button -->
		{#if !isMobile}
			<Button
				aria-label="Close details"
				class="size-7"
				onclick={onClose}
				size="icon"
				variant="ghost"
			>
				<X class="size-icon-md" />
			</Button>
		{/if}
	</div>

	<!-- a phone floats the actions at the bottom of the pane, where the thumb is -->
	{#if !isMobile}
		<ModelsManagerModelConfigurationActions
			{isDownload}
			{isDownloadPaused}
			{isLoaded}
			{isLoading}
			{onRequestCancelDownload}
			{onSeeInDiscover}
			{onToggleDownload}
			{onToggleLoad}
			{onUseInChat}
			{onUseInNewChat}
			size="default"
		/>
	{/if}
</header>
