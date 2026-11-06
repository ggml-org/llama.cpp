<!--
	A piece with buttons under it to download it as WebP or PNG, like the brand's asset files, but rendered from the
	page as it stands (its copy and ground as set in the controls) at the piece's real size, by modern-screenshot. Its
	fonts are embedded, and a same-origin scene in an iframe (the Store's quick chat) comes along.
-->
<script lang="ts">
	import { Download, LoaderCircle } from '@lucide/svelte';
	import { Button } from '$lib/components/ui/button';
	import { domToPng, domToWebp } from 'modern-screenshot';
	import type { Snippet } from 'svelte';

	interface Props {
		children: Snippet;
		/** the file's name, without its extension */
		name: string;
	}

	let { children, name }: Props = $props();

	type Format = 'png' | 'webp';

	const formats: { format: Format; label: string }[] = [
		{ format: 'webp', label: 'WebP' },
		{ format: 'png', label: 'PNG' }
	];

	let box = $state<HTMLDivElement>();
	let busy = $state<Format | null>(null);

	async function download(format: Format) {
		// the piece, drawn at its real size under the scale that fits it to the column
		const piece = box?.querySelector<HTMLElement>('article.post');

		if (!piece || busy) return;

		busy = format;

		try {
			const options = {
				height: piece.offsetHeight,
				scale: 1,
				style: { transform: 'none' },
				width: piece.offsetWidth
			};
			const src =
				format === 'png'
					? await domToPng(piece, options)
					: await domToWebp(piece, { ...options, quality: 0.95 });
			// a link that is never in the page, so its click doesn't reach the window, where @storybook/sveltekit's
			// MockProvider cancels link clicks to log them as navigation
			const link = document.createElement('a');

			link.download = `${name}.${format}`;
			link.href = src;
			link.click();
		} finally {
			busy = null;
		}
	}
</script>

<div bind:this={box} class="grid gap-2">
	{@render children()}

	<span class="flex flex-wrap gap-1">
		{#each formats as { format, label } (format)}
			<Button
				class="h-6 px-1.5 font-mono text-xs text-muted-foreground has-[>svg]:px-1.5"
				disabled={busy !== null}
				onclick={() => download(format)}
				size="sm"
				variant="ghost"
			>
				{#if busy === format}<LoaderCircle class="size-3 animate-spin" />{:else}<Download
						class="size-3"
					/>{/if}{label}
			</Button>
		{/each}
	</span>
</div>
