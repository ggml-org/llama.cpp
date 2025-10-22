<script lang="ts">
	import * as Dialog from '$lib/components/ui/dialog';

	interface Props {
		open: boolean;
		code: string;
		language: string;
		onOpenChange?: (open: boolean) => void;
	}

	let { open = $bindable(), code, language, onOpenChange }: Props = $props();

	let iframeRef = $state<HTMLIFrameElement | null>(null);

	$effect(() => {
		if (iframeRef) {
			if (open) {
				iframeRef.srcdoc = code;
			} else {
				iframeRef.srcdoc = '';
			}
		}
	});

	function handleOpenChange(nextOpen: boolean) {
		open = nextOpen;
		onOpenChange?.(nextOpen);
	}
</script>

<Dialog.Root {open} onOpenChange={handleOpenChange}>
	<Dialog.Content class="max-w-[calc(100%-1rem)] sm:max-w-4xl md:max-w-5xl">
		<Dialog.Header>
			<Dialog.Title>HTML Preview</Dialog.Title>
		</Dialog.Header>

		<div class="preview-container mt-2">
			<iframe
				bind:this={iframeRef}
				title={`Preview ${language}`}
				sandbox="allow-scripts"
				class="h-[70vh] w-full rounded-md border border-border/40 bg-background"
			></iframe>
		</div>
	</Dialog.Content>
</Dialog.Root>

<style>
	.preview-container {
		display: flex;
		flex-direction: column;
		gap: 0.75rem;
	}
</style>
