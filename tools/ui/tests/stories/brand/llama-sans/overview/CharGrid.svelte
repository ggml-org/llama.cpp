<!-- The character set in a hairline grid, after github.com/mona-sans: capitals, lowercase, figures and punctuation. -->
<script lang="ts">
	const chars = [
		...'ABCDEFGHIJKLMNOPQRSTUVWXYZ',
		...'abcdefghijklmnopqrstuvwxyz',
		...'0123456789',
		...'!?#%&()[]{}/@*+=<>.,:;’“”–',
		''
	];
</script>

<section aria-label="Character set" class="chars">
	{#each chars as ch (ch)}<span>{ch}</span>{/each}
</section>

<style>
	.chars {
		display: grid;
		grid-template-columns: repeat(auto-fill, minmax(clamp(48px, 7vw, 84px), 1fr));
		width: var(--column);
		margin-inline: auto;
		border-top: 1px solid var(--rule);
		border-left: 1px solid var(--rule);
	}
	span {
		display: grid;
		place-items: center;
		aspect-ratio: 1;
		border-right: 1px solid var(--rule);
		border-bottom: 1px solid var(--rule);
		font: 400 clamp(22px, 3vw, 36px) / 1 var(--ui);
	}
</style>
