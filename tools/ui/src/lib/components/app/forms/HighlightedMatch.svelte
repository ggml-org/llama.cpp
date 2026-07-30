<script lang="ts">
	/**
	 * Splits `text` into alternating non-match/match segments at each
	 * case-insensitive occurrence of `query`. Used by picker rows to
	 * highlight the user's search term inside longer path strings.
	 *
	 * Pure function - exported in case a future consumer needs to inspect
	 * the segments rather than render them through the default markup.
	 */
	export function splitMatch(text: string, query: string): { text: string; match: boolean }[] {
		if (!query) return [{ text, match: false }];
		const segments: { text: string; match: boolean }[] = [];
		const lowerText = text.toLowerCase();
		const lowerQuery = query.toLowerCase();
		let i = 0;
		while (i < text.length) {
			const idx = lowerText.indexOf(lowerQuery, i);
			if (idx < 0) {
				segments.push({ text: text.slice(i), match: false });
				break;
			}
			if (idx > i) segments.push({ text: text.slice(i, idx), match: false });
			segments.push({ text: text.slice(idx, idx + query.length), match: true });
			i = idx + query.length;
		}
		return segments;
	}

	interface Props {
		text: string;
		query: string;
		matchClass?: string;
	}

	let {
		text,
		query,
		matchClass = 'rounded bg-yellow-200/60 px-0.5 text-foreground dark:bg-yellow-500/30'
	}: Props = $props();

	let segments = $derived(splitMatch(text, query));
</script>

{#each segments as seg, i (i)}
	{#if seg.match}
		<mark class={matchClass}>{seg.text}</mark>
	{:else}
		{seg.text}
	{/if}
{/each}
