<script lang="ts">
	import XIcon from '@lucide/svelte/icons/x';
	import { Dialog as DialogPrimitive } from 'bits-ui';

	interface Props {
		open: boolean;
		code: string;
		language: string;
		onOpenChange?: (open: boolean) => void;
	}

	let { code, language, onOpenChange, open = $bindable() }: Props = $props();

	let iframeRef = $state<HTMLIFrameElement | null>(null);

	$effect(() => {
		if (!iframeRef) return;

		if (open) {
			iframeRef.srcdoc = code;
		} else {
			iframeRef.srcdoc = '';
		}
	});

	function handleOpenChange(nextOpen: boolean) {
		open = nextOpen;

		onOpenChange?.(nextOpen);
	}
</script>

<DialogPrimitive.Root onOpenChange={handleOpenChange} {open}>
	<DialogPrimitive.Portal>
		<DialogPrimitive.Overlay class="code-preview-overlay" />

		<DialogPrimitive.Content class="code-preview-content">
			<iframe
				bind:this={iframeRef}
				class="code-preview-iframe"
				sandbox="allow-scripts"
				title="Preview {language}"
			></iframe>

			<DialogPrimitive.Close
				aria-label="Close preview"
				class="code-preview-close absolute top-4 right-4 border-none bg-transparent text-white opacity-70 mix-blend-difference transition-opacity hover:opacity-100 focus-visible:ring-0 focus-visible:ring-offset-0 focus-visible:outline-none disabled:pointer-events-none [&_svg]:pointer-events-none [&_svg]:shrink-0 [&_svg:not([class*='size-'])]:size-8"
			>
				<XIcon />

				<span class="sr-only">Close preview</span>
			</DialogPrimitive.Close>
		</DialogPrimitive.Content>
	</DialogPrimitive.Portal>
</DialogPrimitive.Root>

<style lang="postcss">
	:global(.code-preview-overlay) {
		position: fixed;
		inset: 0;
		background-color: transparent;
		z-index: 100000;
	}

	:global(.code-preview-content) {
		position: fixed;
		inset: 0;
		top: 0 !important;
		left: 0 !important;
		width: 100dvw;
		height: 100dvh;
		margin: 0;
		padding: 0;
		border: none;
		border-radius: 0;
		background-color: transparent;
		box-shadow: none;
		display: block;
		overflow: hidden;
		transform: none !important;
		z-index: 100001;
	}

	:global(.code-preview-iframe) {
		display: block;
		width: 100dvw;
		height: 100dvh;
		border: 0;
	}

	:global(.code-preview-close) {
		position: absolute;
		z-index: 100002;
	}
</style>
