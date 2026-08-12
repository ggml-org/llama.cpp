<script lang="ts">
	import MentionBadge from './MentionBadge.svelte';
	import { splitMentionSegments } from '$lib/utils';

	interface Props {
		content: string;
	}

	let { content }: Props = $props();

	let segments = $derived(splitMentionSegments(content));
</script>

<!-- Segments sit in a `whitespace-pre-wrap` parent, so the markup stays
     glued: any newline between the tags below would print as a space. -->
<!-- prettier-ignore -->
{#each segments as segment, index (index)}{#if segment.mention}<MentionBadge name={segment.mention.name} path={segment.mention.path} />{:else}{segment.text}{/if}{/each}
