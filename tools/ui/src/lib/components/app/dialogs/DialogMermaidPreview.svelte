<script lang="ts">
	import { Dialog as DialogPrimitive } from 'bits-ui';
	import XIcon from '@lucide/svelte/icons/x';
	import MinusIcon from '@lucide/svelte/icons/minus';
	import PlusIcon from '@lucide/svelte/icons/plus';
	import DownloadIcon from '@lucide/svelte/icons/download';
	import { mode } from 'mode-watcher';
	import { ColorMode } from '$lib/enums';

	interface Props {
		open: boolean;
		code: string;
		onOpenChange?: (open: boolean) => void;
	}

	let { open = $bindable(), code, onOpenChange }: Props = $props();

	let svg = $state('');
	let error = $state<string | null>(null);
	let containerRef = $state<HTMLDivElement | null>(null);
	let scale = $state(1);
	let panX = $state(0);
	let panY = $state(0);
	let isDragging = $state(false);
	let lastMouseX = 0;
	let lastMouseY = 0;

	const ZOOM_STEP = 0.5;
	const ZOOM_MIN = 0.5;
	const ZOOM_MAX = 5;

	$effect(() => {
		if (open) {
			function onGlobalMouseUp() {
				isDragging = false;
			}
			document.addEventListener('mouseup', onGlobalMouseUp);
			return () => document.removeEventListener('mouseup', onGlobalMouseUp);
		}
	});

	$effect(() => {
		if (!open || !code) {
			svg = '';
			error = null;
			scale = 1;
			resetPan();
			return;
		}

		async function render() {
			svg = '';
			error = null;
			scale = 1;
			resetPan();

			try {
				const { default: mermaid } = await import('mermaid');

				const isDark = mode.current === ColorMode.DARK;
				mermaid.initialize({
					startOnLoad: false,
					theme: isDark ? 'dark' : 'default',
					securityLevel: 'strict'
				});

				const id = `mermaid-preview-${Date.now()}`;
				const result = await mermaid.render(id, code);
				svg = result.svg.replace(/ width="[^"]*"/, '').replace(/ height="[^"]*"/, '');

				if (result.bindFunctions && containerRef) {
					result.bindFunctions(containerRef);
				}
			} catch (err) {
				error = err instanceof Error ? err.message : 'Failed to render diagram';
				console.error('Failed to render mermaid diagram:', err);
			}
		}

		render();
	});

	function zoomIn() {
		scale = Math.min(scale + ZOOM_STEP, ZOOM_MAX);
		resetPan();
	}

	function zoomOut() {
		scale = Math.max(scale - ZOOM_STEP, ZOOM_MIN);
		resetPan();
	}

	function downloadSvg() {
		const blob = new Blob([svg], { type: 'image/svg+xml' });
		const url = URL.createObjectURL(blob);
		const a = document.createElement('a');
		a.href = url;
		a.download = 'mermaid-diagram.svg';
		a.click();
		URL.revokeObjectURL(url);
	}

	function handleMouseDown(e: MouseEvent) {
		if (scale <= 1) return;
		isDragging = true;
		lastMouseX = e.clientX;
		lastMouseY = e.clientY;
	}

	function handleMouseMove(e: MouseEvent) {
		if (!isDragging) return;
		panX += e.clientX - lastMouseX;
		panY += e.clientY - lastMouseY;
		lastMouseX = e.clientX;
		lastMouseY = e.clientY;
	}

	function handleMouseUp() {
		isDragging = false;
	}

	function resetPan() {
		panX = 0;
		panY = 0;
	}

	function handleOpenChange(nextOpen: boolean) {
		open = nextOpen;
		onOpenChange?.(nextOpen);
	}
</script>

<DialogPrimitive.Root {open} onOpenChange={handleOpenChange}>
	<DialogPrimitive.Portal>
		<DialogPrimitive.Overlay class="mermaid-preview-overlay" />

		<DialogPrimitive.Content class="mermaid-preview-content">
			{#if error}
				<div class="mermaid-preview-error">
					<p class="error-title">Failed to render diagram</p>
					<p class="error-message">{error}</p>
				</div>
			{:else}
				<!-- svelte-ignore a11y_no_static_element_interactions -->
				<div
					bind:this={containerRef}
					class="mermaid-preview-container"
					style="transform: scale({scale}) translate({panX}px, {panY}px); cursor: {scale > 1
						? 'grab'
						: 'default'};"
					onmousedown={handleMouseDown}
					onmousemove={handleMouseMove}
					onmouseup={handleMouseUp}
				>
					{@html svg}
				</div>

				<div class="mermaid-preview-zoom">
					<button class="zoom-btn" onclick={zoomOut} title="Zoom out">
						<MinusIcon class="zoom-icon" />
					</button>
					<span class="zoom-level">{Math.round(scale * 100)}%</span>
					<button class="zoom-btn" onclick={zoomIn} title="Zoom in">
						<PlusIcon class="zoom-icon" />
					</button>
					<div class="zoom-divider"></div>
					<button class="zoom-btn" onclick={downloadSvg} title="Download SVG">
						<DownloadIcon class="zoom-icon" />
					</button>
				</div>
			{/if}

			<DialogPrimitive.Close
				class="mermaid-preview-close absolute top-4 right-4 border-none bg-transparent text-white opacity-70 mix-blend-difference transition-opacity hover:opacity-100 focus-visible:ring-0 focus-visible:ring-offset-0 focus-visible:outline-none disabled:pointer-events-none [&_svg]:pointer-events-none [&_svg]:shrink-0 [&_svg:not([class*='size-'])]:size-8"
				aria-label="Close preview"
			>
				<XIcon />

				<span class="sr-only">Close preview</span>
			</DialogPrimitive.Close>
		</DialogPrimitive.Content>
	</DialogPrimitive.Portal>
</DialogPrimitive.Root>

<style lang="postcss">
	:global(.mermaid-preview-overlay) {
		position: fixed;
		inset: 0;
		background-color: rgba(0, 0, 0, 0.75);
		z-index: 100000;
	}

	:global(.mermaid-preview-content) {
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
		background-color: var(--background);
		box-shadow: none;
		display: flex;
		align-items: center;
		justify-content: center;
		overflow: auto;
		transform: none !important;
		z-index: 100001;
	}

	:global(.mermaid-preview-container) {
		display: flex;
		align-items: center;
		justify-content: center;
		padding: 2rem;
		flex: 1;
		max-width: 90vw;
		max-height: 90dvh;
		user-select: none;
	}

	:global(.mermaid-preview-container svg) {
		max-width: 100%;
		max-height: calc(100dvh - 4rem);
		width: 100%;
		height: 100%;
		display: block;
	}

	:global(.mermaid-preview-zoom) {
		position: fixed;
		top: 1rem;
		left: 50%;
		transform: translateX(-50%);
		display: flex;
		align-items: center;
		gap: 0.75rem;
		padding: 0.5rem 1rem;
		background: rgba(0, 0, 0, 0.6);
		border-radius: 0.5rem;
		z-index: 100002;
	}

	:global(.zoom-btn) {
		display: flex;
		align-items: center;
		justify-content: center;
		padding: 0.25rem;
		background: transparent;
		border: none;
		color: white;
		cursor: pointer;
		opacity: 0.7;
		transition: opacity 0.2s ease;
	}

	:global(.zoom-btn:hover) {
		opacity: 1;
	}

	:global(.zoom-icon) {
		width: 18px;
		height: 18px;
	}

	:global(.zoom-level) {
		font-size: 0.875rem;
		font-weight: 500;
		color: white;
		min-width: 3ch;
		text-align: center;
	}

	:global(.zoom-divider) {
		width: 1px;
		height: 16px;
		background: rgba(255, 255, 255, 0.3);
	}

	:global(.mermaid-preview-close) {
		position: absolute;
		z-index: 100002;
	}

	:global(.mermaid-preview-error) {
		display: flex;
		flex-direction: column;
		align-items: center;
		justify-content: center;
		gap: 1rem;
		padding: 2rem;
		text-align: center;
		color: white;
	}

	:global(.error-title) {
		font-size: 1.25rem;
		font-weight: 600;
		margin: 0;
	}

	:global(.error-message) {
		font-size: 0.875rem;
		opacity: 0.7;
		margin: 0;
		max-width: 400px;
	}
</style>
