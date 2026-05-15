<script lang="ts">
	import { Music, FileText } from '@lucide/svelte';
	import { HorizontalScrollCarousel } from '$lib/components/app/misc';

	interface PreviewItem {
		id: string;
		name: string;
		isImage: boolean;
		isAudio: boolean;
		preview?: string;
	}

	interface Props {
		items: PreviewItem[];
		currentIndex: number;
		onNavigate: (index: number) => void;
	}

	let { items, currentIndex, onNavigate }: Props = $props();

	function getFileExtension(name: string): string {
		const parts = name.split('.');
		if (parts.length > 1) {
			return parts.pop()?.toUpperCase() ?? '';
		}
		return '';
	}
</script>

{#if items.length > 1}
	<div class="sticky bottom-0 z-10 mt-4 flex-shrink-0">
		<HorizontalScrollCarousel class="max-w-full">
			{#each items as item, index (item.id)}
				<button
					data-thumbnail-index={index}
					class={[
						'relative flex-shrink-0 cursor-pointer overflow-hidden rounded border-2 bg-black/80 backdrop-blur-sm transition-all hover:opacity-90',
						index === currentIndex ? 'border-white' : 'border-transparent opacity-60',
						'[&:not(:first-child)]:last:mr-4 [&:not(:last-child)]:first:ml-4'
					]}
					onclick={() => onNavigate(index)}
					aria-label={`Go to ${item.name}`}
				>
					{#if item.isImage && item.preview}
						<img src={item.preview} alt={item.name} class="h-12 w-12 object-cover" />
					{:else}
						<div
							class="bg-foreground-muted/50 flex h-12 w-12 flex-col items-center justify-center gap-0.5 py-1"
						>
							{#if item.isAudio}
								<Music class="h-4 w-4 text-white/70" />
							{:else}
								<FileText class="h-4 w-4 text-white/70" />
							{/if}

							<span class="font-mono text-[9px] text-white/60">{getFileExtension(item.name)}</span>
						</div>
					{/if}
				</button>
			{/each}
		</HorizontalScrollCarousel>
	</div>
{/if}
