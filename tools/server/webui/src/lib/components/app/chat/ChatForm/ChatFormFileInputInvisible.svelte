<script lang="ts">
	interface Props {
		class?: string;
		multiple?: boolean;
		onFileSelect?: (files: File[]) => void;
	}

	let { class: className = '', multiple = true, onFileSelect }: Props = $props();

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
	onchange={handleFileSelect}
	class="hidden {className}"
/>
