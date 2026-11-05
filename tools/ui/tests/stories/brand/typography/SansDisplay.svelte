<!--
	Llama Sans Display's part of the Typography overview, after its hero: what it's for, what sets it apart
	(highlights), headings and posters set in it, the numbers, the characters and weights, and the shared design notes.
-->
<script lang="ts">
	import Metrics from '../llama-sans/lib/components/Metrics.svelte';
	import CharGrid from '../llama-sans/overview/CharGrid.svelte';
	import DisplaySample from '../llama-sans/overview/DisplaySample.svelte';
	import Posters from '../llama-sans/overview/Posters.svelte';
	import Section from '../llama-sans/overview/Section.svelte';
	import Version from '../llama-sans/overview/Version.svelte';
	import WeightLines from '../llama-sans/overview/WeightLines.svelte';
	import Highlights, { type Highlight } from './Highlights.svelte';
	import SansNotes from './SansNotes.svelte';

	const face = 'var(--headline)';

	const highlights: Highlight[] = [
		{
			sample: 'Local inference',
			spec: 'Tracking −22',
			text: 'Set tighter than UI, so large words hold together as one shape instead of falling apart into letters.',
			title: 'Tight at large sizes'
		},
		{
			sample: 'AVAT Type',
			spec: 'Kerning 115%',
			text: 'Every kerning pair is a little firmer than UI’s, because gaps that pass at 14 px show at 64.',
			title: 'Firmer kerning'
		},
		{
			sample: 'the thinking',
			spec: 'th +10',
			text: 'With the tighter tracking, t leaves room before b, h, k and l, so th and tl don’t touch in headlines.',
			title: 'Room after the t'
		},
		{
			sample: 'Two lines,\none heading',
			spec: 'Line 1.15',
			text: 'A compact line, so a heading that wraps still reads as one block.',
			title: 'Compact lines'
		},
		{
			sample: 'Aa  Aa',
			spec: 'Shared outlines',
			text: 'The same letters as UI, so a heading and the paragraph under it are visibly one family; only the spacing changes.',
			title: 'One family with UI'
		},
		{
			sample: 'h1 – h6',
			spec: '--font-display',
			text: 'The design system sets every heading, from the chat greeting down to h6, in Display.',
			title: 'Every heading'
		}
	];
</script>

<Section columns label="For headlines and large type">
	<p>
		Llama Sans Display is for headings, titles, posters and anything set large. It shares its
		outlines with UI and tightens what large sizes need tightened: tracking, kerning and the line.
	</p>

	<p>Use it from about 24 px up. Below that, Llama Sans UI's looser spacing reads better.</p>
</Section>

<Section label="Highlights">
	<Highlights {face} items={highlights} large />
</Section>

<Version
	name="Display"
	side="end"
	specs="Tracking −22 · kerning 115% · line 1.15"
	statement="Quiet letters for loud ideas."
>
	<DisplaySample />
</Version>

<Posters />

<Section label="Numbers">
	<Metrics />
</Section>

<CharGrid {face} />

<WeightLines {face} />

<SansNotes />
