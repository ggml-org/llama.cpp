<script lang="ts">
	import * as Dialog from '$lib/components/ui/dialog/index.js';
	import { MermaidPreview } from '$lib/components/app/content';

	interface Props {
		open: boolean;
		svgHtml: string;
		onOpenChange?: (open: boolean) => void;
	}

	let { open = $bindable(), svgHtml, onOpenChange }: Props = $props();
</script>

<Dialog.Root bind:open {onOpenChange}>
	<Dialog.Content
		class="z-999999 grid max-h-full max-w-full! grid-rows-[1fr_auto] overflow-hidden p-0 md:h-[90vh] md:max-w-[90vw]!"
	>
		<MermaidPreview {svgHtml} />
	</Dialog.Content>
</Dialog.Root>
