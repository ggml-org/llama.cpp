<!--
	Text that is being spoken: what has been said so far in the full foreground colour, which it keeps, and the rest
	muted. Used for a transcript followed during playback (`spoken`, e.g. from spokenLength()), for a reply read aloud,
	and for text that streams in while someone speaks (`live`: each new word brightens as it arrives).
	Without `spoken` or `live` all of it reads as spoken. The words are plain text for screen readers.
-->
<script lang="ts">
	import { cn } from '../ui/utils';
	import { spokenWords } from '../../voice/spoken';

	interface Props {
		text: string;
		/** Characters spoken so far; left out, all of them. */
		spoken?: number | null;
		/** Text arriving as it is said: new words come in muted and brighten. */
		live?: boolean;
		class?: string;
	}

	let { class: className, live = false, spoken = null, text }: Props = $props();

	let words = $derived(spokenWords(text));
	let reached = $derived(spoken ?? text.length);
</script>

<span class={cn('whitespace-pre-wrap', className)} data-slot="voice-spoken-text"
	>{#each words as { start, word }, index (live ? index : start)}<span
			class={[
				'transition-colors duration-300 ease-out motion-reduce:transition-none',
				start < reached ? 'text-foreground' : 'text-muted-foreground',
				live && 'spoken-in'
			]}
			data-spoken={start < reached || undefined}>{word}</span
		>{/each}</span
>

<style>
	@keyframes spoken-in {
		from {
			color: var(--fg-muted, currentColor);
		}
	}

	.spoken-in {
		animation: spoken-in 400ms ease-out;
	}

	@media (prefers-reduced-motion: reduce) {
		.spoken-in {
			animation: none;
		}
	}
</style>
