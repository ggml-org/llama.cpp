<!-- The design notes, the next pass and the licence that Llama Sans UI and Display share, at the end of each one's part. -->
<script lang="ts">
	import Note from '../llama-sans/lib/components/Note.svelte';
	import Notes from '../llama-sans/lib/components/Notes.svelte';
	import { specimen } from '../llama-sans/lib/specimen';
	import Section from '../llama-sans/overview/Section.svelte';

	const stems = specimen.styles.map((s) => s.stem);
</script>

<Section label="Design notes">
	<Notes>
		<Note term="Stroke"
			>Near-monoline, with a {stems[0]}-unit stem at 400 rising to {stems.at(-1)} at 700. Where a bowl,
			arm or shoulder leaves a stem, the notch cuts a little deeper into the joint, so joins stay open
			without turning into ink traps.</Note
		>

		<Note term="Curves"
			>Bowls are eased toward true circles, fully circular in G, S and s, so rounds read soft and
			calm without turning geometric.</Note
		>

		<Note term="Terminals"
			>Every curved stroke end is trimmed the same short distance and cut almost square to the
			stroke, which keeps c, e, g, s, 2, 3, 5, 6 and 9 open and consistent.</Note
		>

		<Note term="a"
			>Double-story, on a straight stem with a flat foot. The hook runs long and ends in a flat cut;
			the bowl rises gently into the stem with sharp joints. Toward Bold, a small bracketed foot
			appears.</Note
		>

		<Note term="l, r, t, f"
			>A plain l the width of the i, and a narrow t with short arms. The ends of r, t and f are
			angled to match the curved terminals.</Note
		>

		<Note term="G, S, s"
			>G and S wide and round, s a little leaner, with slightly open, angled ends on S and s.</Note
		>

		<Note term="Weights"
			>Regular, Medium, SemiBold and Bold at 400, 500, 600 and 700, a little lighter than the
			numbers usually mean, with every weight in between.</Note
		>
	</Notes>
</Section>

<Section label="Next pass">
	<p>
		A straight-legged R and a shorter Q tail, an angled end on the j, slightly wider o, c, e and O,
		and a review of the accented glyphs.
	</p>
</Section>

<Section label="Free & open source">
	<p>
		Llama Sans is released under the SIL Open Font License 1.1. It is derived from Geist, © 2024 The
		Geist Project Authors. The Llama icon at U+E000 is the ggml-org mark, licensed separately under
		CC BY-NC 4.0.
	</p>
</Section>
