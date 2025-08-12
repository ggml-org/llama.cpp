<script lang="ts">
	import { ALL_SUPPORTED_EXTENSIONS, ALL_SUPPORTED_MIME_TYPES } from '$lib/constants/supported-file-types';
	import { generateModalityAwareAcceptString } from '$lib/utils/modality-file-validation';

	interface Props {
		accept?: string;
		multiple?: boolean;
		onFileSelect?: (files: File[]) => void;
		class?: string;
	}

	// Generate accept string from our enum-based supported file types
	const defaultAccept = [
		...ALL_SUPPORTED_EXTENSIONS,
		...ALL_SUPPORTED_MIME_TYPES,
	].join(',');

	let {
		accept,
		multiple = true,
		onFileSelect,
		class: className = ''
	}: Props = $props();

	// Use modality-aware accept string by default, but allow override
	const finalAccept = $derived(accept ?? generateModalityAwareAcceptString());

	let fileInputElement: HTMLInputElement | undefined;

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
