<script lang="ts">
	import * as AlertDialog from '@ggml-org/llama-ui/ui/alert-dialog';
	import { Button } from '@ggml-org/llama-ui/ui/button';
	import { RotateCcw } from '@lucide/svelte';
	import { settingsStore } from '#lib/stores/index.ts';

	interface Props {
		onReset?: () => void;
	}

	let { onReset }: Props = $props();

	let showResetDialog = $state(false);

	function handleResetClick() {
		showResetDialog = true;
	}

	function handleConfirmReset() {
		settingsStore.forceSyncWithServerDefaults();
		onReset?.();

		showResetDialog = false;
	}
</script>

{#if settingsStore.hasCustomSettings}
	<Button onclick={handleResetClick} variant="outline">
		<RotateCcw class="size-icon-md" />

		Reset to default
	</Button>
{/if}

<AlertDialog.Root bind:open={showResetDialog}>
	<AlertDialog.Content>
		<AlertDialog.Header>
			<AlertDialog.Title>Reset Settings to Default</AlertDialog.Title>

			<AlertDialog.Description>
				Are you sure you want to reset all settings to their default values? This will reset all
				parameters to the values provided by the server's /props endpoint and remove all your custom
				configurations.
			</AlertDialog.Description>
		</AlertDialog.Header>

		<AlertDialog.Footer>
			<AlertDialog.Cancel>Cancel</AlertDialog.Cancel>

			<AlertDialog.Action onclick={handleConfirmReset}>Reset to Default</AlertDialog.Action>
		</AlertDialog.Footer>
	</AlertDialog.Content>
</AlertDialog.Root>
