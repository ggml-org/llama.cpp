<script lang="ts">
	import { Download } from '@lucide/svelte';
	import ZoomInIcon from '@lucide/svelte/icons/zoom-in';
	import ZoomOutIcon from '@lucide/svelte/icons/zoom-out';
	import RotateCcwIcon from '@lucide/svelte/icons/rotate-ccw';

	interface Props {
		scale: number;
		svgHtml: string;
		onZoomIn: () => void;
		onZoomOut: () => void;
		onResetView: () => void;
	}

	let { scale, svgHtml, onZoomIn, onZoomOut, onResetView }: Props = $props();

	function downloadSvg() {
		if (!svgHtml) return;
		const blob = new Blob([svgHtml], { type: 'image/svg+xml' });
		const url = URL.createObjectURL(blob);
		const a = document.createElement('a');
		a.href = url;
		a.download = 'diagram.svg';
		a.click();
		URL.revokeObjectURL(url);
	}
</script>

<div
	class="mermaid-preview-controls absolute bottom-8 flex shrink-0 items-center justify-center p-3"
>
	<div class="mermaid-preview-controls-inner flex items-center gap-1 rounded-lg bg-muted p-1">
		<button
			class="mermaid-preview-btn flex h-8 w-8 cursor-pointer items-center justify-center rounded-md border-0 bg-transparent text-foreground transition-colors hover:bg-muted-foreground/15 active:bg-muted-foreground/25"
			onclick={onZoomOut}
			title="Zoom out"
			aria-label="Zoom out"
		>
			<ZoomOutIcon class="mermaid-preview-btn-icon h-4 w-4" />
		</button>
		<span
			class="mermaid-preview-zoom-label min-w-[3.5rem] px-0.5 text-center text-xs font-medium text-muted-foreground tabular-nums select-none"
			>{Math.round(scale * 100)}%</span
		>
		<button
			class="mermaid-preview-btn flex h-8 w-8 cursor-pointer items-center justify-center rounded-md border-0 bg-transparent text-foreground transition-colors hover:bg-muted-foreground/15 active:bg-muted-foreground/25"
			onclick={onZoomIn}
			title="Zoom in"
			aria-label="Zoom in"
		>
			<ZoomInIcon class="mermaid-preview-btn-icon h-4 w-4" />
		</button>
		<div class="mermaid-preview-controls-separator mx-1 h-5 w-px bg-border/50"></div>

		<button
			class="mermaid-preview-btn flex h-8 w-8 cursor-pointer items-center justify-center rounded-md border-0 bg-transparent text-foreground transition-colors hover:bg-muted-foreground/15 active:bg-muted-foreground/25"
			onclick={onResetView}
			title="Reset view"
			aria-label="Reset view"
		>
			<RotateCcwIcon class="mermaid-preview-btn-icon h-4 w-4" />
		</button>
		<div class="mermaid-preview-controls-separator mx-1 h-5 w-px bg-border/50"></div>

		<button
			class="mermaid-preview-btn flex h-8 w-8 cursor-pointer items-center justify-center rounded-md border-0 bg-transparent text-foreground transition-colors hover:bg-muted-foreground/15 active:bg-muted-foreground/25"
			onclick={downloadSvg}
			title="Download SVG"
			aria-label="Download SVG"
		>
			<Download class="mermaid-preview-btn-icon h-4 w-4" />
		</button>
	</div>
</div>
