<!--
	Llama Sans Display's part of the Typography overview, after its hero: what it's for, what sets it apart
	(highlights), the optical size axis, headings and posters set in it, the numbers, the characters and weights, and
	the shared design notes.
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
	import OpticalSizes from './OpticalSizes.svelte';
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
			spec: 'line-height 1.15',
			text: 'Headings set a compact 1.15 line, so one that wraps still reads as one block. The font keeps UI’s cap-centred line box at every size.',
			title: 'Compact lines'
		},
		{
			sample: 'Aa  Aa',
			spec: 'One font · opsz 16–40',
			text: 'The same font as UI, so a heading and the paragraph under it are visibly one family; only the spacing changes, with the size.',
			title: 'One font with UI'
		},
		{
			sample: 'h1 – h6',
			spec: 'font-optical-sizing: auto',
			text: 'Headings are Llama Sans like everything else; their size alone moves them toward Display, all the way from 40 px.',
			title: 'Every large heading'
		}
	];
</script>

<Section columns label="For headlines and large type">
	<p>
		Llama Sans Display is Llama Sans at its large optical size, opsz 40: headings, titles, posters
		and anything set large, with the tracking and kerning large sizes need.
	</p>

	<p>
		There's no separate font to pick: from 16 to 40 px the spacing moves on its own from UI's to
		Display's, and from 40 px up it's all Display.
	</p>
</Section>

<Section label="Highlights">
	<Highlights {face} items={highlights} large />
</Section>

<Section label="Optical size">
	<OpticalSizes />
</Section>

<Version
	name="Display"
	side="end"
	specs="opsz 40 · tracking −22 · kerning 115% · line-height 1.15"
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
