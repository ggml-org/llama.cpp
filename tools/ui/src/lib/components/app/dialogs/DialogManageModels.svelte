<script lang="ts">
	import { Logo } from '$lib/components/app/misc';
	import { ModelsDiscover } from '$lib/components/app/models/discover';
	import { Button } from '$lib/components/ui/button';
	import * as Dialog from '$lib/components/ui/dialog';
	import { SETTINGS_KEYS } from '$lib/constants';
	import { serverStore, settingsStore } from '$lib/stores';

	interface Props {
		open?: boolean;
		onOpenChange?: (open: boolean) => void;
	}

	let { onOpenChange, open = $bindable(false) }: Props = $props();

	// discovery needs the local router's download endpoints, and the user's opt-in
	let isEnabled = $derived(settingsStore.config[SETTINGS_KEYS.ENABLE_DISCOVER_MODELS] === true);
	let hasDiscover = $derived(serverStore.localIsRouter && isEnabled);

	function handleOpenChange(value: boolean) {
		open = value;
		onOpenChange?.(value);
	}
</script>

<Dialog.Root onOpenChange={handleOpenChange} {open}>
	<Dialog.Content
		class="flex flex-col md:h-[calc(100vh-4rem)]! md:max-h-240! md:w-[calc(100vw-4rem)]! md:max-w-360! md:overflow-hidden"
	>
		<Dialog.Header>
			<Dialog.Title class="flex items-center gap-2">
				<Logo class="h-5 w-5" style="--size: 1.25rem" />

				<span>Discover models</span>
			</Dialog.Title>

			<Dialog.Description
				>Download models from the catalog into the local server.</Dialog.Description
			>
		</Dialog.Header>

		{#if hasDiscover}
			<div class="grid min-h-0 flex-1 overflow-hidden" style="grid-template-columns: auto 1fr;">
				<ModelsDiscover />
			</div>
		{:else}
			<div class="flex min-h-0 flex-1 flex-col items-start gap-3 text-sm text-muted-foreground">
				{#if !serverStore.localIsRouter}
					<p>
						Discovery downloads into the local server, which has to run in router mode. Start
						llama-server with <code class="text-foreground">--model-dir</code> to enable it.
					</p>
				{:else}
					<p>Discovery is turned off in settings.</p>

					<Button
						onclick={() => settingsStore.updateConfig(SETTINGS_KEYS.ENABLE_DISCOVER_MODELS, true)}
						variant="outline"
					>
						Enable discovery
					</Button>
				{/if}
			</div>
		{/if}
	</Dialog.Content>
</Dialog.Root>
