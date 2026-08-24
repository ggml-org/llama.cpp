<script lang="ts" module>
	import {
		AudioLines,
		Circle,
		Equal,
		Hash,
		HelpCircle,
		Image as ImageIcon,
		ImagePlus,
		Languages,
		Layers,
		ListCollapse,
		MessageCircle,
		MessageSquare,
		MessageSquareMore,
		Mic,
		Replace,
		Scan,
		Video,
		Volume2
	} from '@lucide/svelte';
	import type { Component } from 'svelte';

	/** Lucide icon name -> component; only pipeline-tag icons we use today. */
	const ICONS: Record<string, Component> = {
		'audio-lines': AudioLines,
		equal: Equal,
		hash: Hash,
		'help-circle': HelpCircle,
		image: ImageIcon,
		'image-plus': ImagePlus,
		languages: Languages,
		layers: Layers,
		'list-collapse': ListCollapse,
		'message-circle': MessageCircle,
		'message-square': MessageSquare,
		'message-square-more': MessageSquareMore,
		mic: Mic,
		replace: Replace,
		scan: Scan,
		video: Video,
		'volume-2': Volume2
	};

	/** Default icon used when the requested name is unknown. */
	const FALLBACK: Component = Circle;
</script>

<script lang="ts">
	interface Props {
		name: string | null;
		class?: string;
	}

	let { class: className = 'h-3.5 w-3.5', name }: Props = $props();

	let Icon = $derived(name && ICONS[name] ? ICONS[name] : FALLBACK);
</script>

<Icon class={className} />
