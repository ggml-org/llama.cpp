<script lang="ts">
	import * as Dialog from '@ggml-org/llama-ui/ui/dialog';
	import { MermaidPreview } from '#lib/components/app/content/index.ts';

	interface Props {
		open: boolean;
		svgHtml: string;
		onOpenChange?: (open: boolean) => void;
	}

	let { onOpenChange, open = $bindable(), svgHtml }: Props = $props();
</script>

<Dialog.Root bind:open {onOpenChange}>
	<Dialog.Content
		class="z-999999 grid max-h-full max-w-full! grid-rows-[1fr_auto] overflow-hidden p-0 md:h-[90vh] md:max-w-[90vw]!"
		showCloseButton
	>
		<MermaidPreview {svgHtml} />
	</Dialog.Content>
</Dialog.Root>
