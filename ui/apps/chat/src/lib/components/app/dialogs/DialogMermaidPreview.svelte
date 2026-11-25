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

<!-- Full screen on every device, so a large diagram has the whole window to pan and zoom in -->
<Dialog.Root bind:open {onOpenChange}>
	<Dialog.Content
		class="top-0! left-0! z-999999 block h-dvh! max-h-none! w-screen! max-w-none! translate-x-0! translate-y-0! overflow-hidden rounded-none! border-x-0! border-t-(length:--safe-area-top)! border-b-(length:--safe-area-bottom)! border-transparent! p-0 data-[state=closed]:zoom-out-100 data-[state=open]:zoom-in-100"
		showCloseButton
	>
		<MermaidPreview {svgHtml} />
	</Dialog.Content>
</Dialog.Root>
