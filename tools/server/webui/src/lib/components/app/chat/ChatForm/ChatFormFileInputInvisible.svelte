<script lang="ts">
	import { generateModalityAwareAcceptString } from '$lib/utils';

	interface Props {
		accept?: string;
		class?: string;
		hasAudioModality?: boolean;
		hasVisionModality?: boolean;
		multiple?: boolean;
		onFileSelect?: (files: File[]) => void;
	}

	let {
		accept = $bindable(),
		class: className = '',
		hasAudioModality = false,
		hasVisionModality = false,
		multiple = true,
		onFileSelect
	}: Props = $props();

	let fileInputElement: HTMLInputElement | undefined;

	// Use modality-aware accept string by default, but allow override
	let finalAccept = $derived(
		accept ??
			generateModalityAwareAcceptString({
				hasVision: hasVisionModality,
				hasAudio: hasAudioModality
			})
	);

	export function click() {
		fileInputElement?.click();
	}

	function handleFileSelect(event: Event) {
		const input = event.target as HTMLInputElement;
		if (input.files) {
			onFileSelect?.(Array.from(input.files));
		}
	}
</script>

<input
	bind:this={fileInputElement}
	type="file"
	{multiple}
	accept={finalAccept}
	onchange={handleFileSelect}
	class="hidden {className}"
/>
