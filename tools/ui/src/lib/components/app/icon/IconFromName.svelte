<script lang="ts" module>
	import type { Component } from 'svelte';
	import {
		AudioLines,
		Circle,
		Equal,
		HelpCircle,
		Hash,
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

	/** Lucide icon name → component; only pipeline-tag icons we use today. */
	const ICONS: Record<string, Component> = {
		equal: Equal,
		'help-circle': HelpCircle,
		hash: Hash,
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
		'volume-2': Volume2,
		'audio-lines': AudioLines
	};

	/** Default icon used when the requested name is unknown. */
	const FALLBACK: Component = Circle;
</script>

<script lang="ts">
	interface Props {
		name: string | null;
		class?: string;
	}

	let { name, class: className = 'h-3.5 w-3.5' }: Props = $props();

	let Icon = $derived(name && ICONS[name] ? ICONS[name] : FALLBACK);
</script>

<Icon class={className} />
