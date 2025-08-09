<script lang="ts">
	interface Props {
		accept?: string;
		multiple?: boolean;
		onFileSelect?: (files: File[]) => void;
		class?: string;
	}

	let {
		accept = "image/*,audio/*,video/*,.pdf,.txt,.doc,.docx",
		multiple = true,
		onFileSelect,
		class: className = ''
	}: Props = $props();

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
	{accept}
	onchange={handleFileSelect}
	class="hidden {className}"
/>
