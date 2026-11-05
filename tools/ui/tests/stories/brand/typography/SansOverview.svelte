<!--
	Llama Sans's part of the Typography overview, after its hero: the font, its details up close and its optical size
	axis, then a part for each end of the axis (UI for interfaces, Display for headlines), each pinned to its opsz with
	its highlights and samples, then the numbers, characters, weights and design notes they share.
-->
<script lang="ts">
	import Metrics from '../llama-sans/lib/components/Metrics.svelte';
	import { specimen } from '../llama-sans/lib/specimen';
	import CharGrid from '../llama-sans/overview/CharGrid.svelte';
	import Closeups from '../llama-sans/overview/Closeups.svelte';
	import DisplaySample from '../llama-sans/overview/DisplaySample.svelte';
	import IconsSample from '../llama-sans/overview/IconsSample.svelte';
	import Posters from '../llama-sans/overview/Posters.svelte';
	import Section from '../llama-sans/overview/Section.svelte';
	import UISample from '../llama-sans/overview/UISample.svelte';
	import WeightLines from '../llama-sans/overview/WeightLines.svelte';
	import Highlights, { type Highlight } from './Highlights.svelte';
	import OpticalSizes from './OpticalSizes.svelte';
	import Part from './Part.svelte';
	import SansNotes from './SansNotes.svelte';

	const { axis, glyphCount, metrics } = specimen;

	const ui: Highlight[] = [
		{
			sample: 'Settings · Models',
			spec: 'Tracking +3',
			text: 'Spaced a touch wider than Display, so labels, menus and captions stay clear at 12 to 16 px instead of clumping.',
			title: 'Spaced for small sizes'
		},
		{
			sample: 'Try Vote Away',
			spec: 'Kerning after SF Pro',
			text: 'Pairs are kerned to match the system fonts at their text sizes, so it sits next to native controls without looking off.',
			title: 'Kerned like the system'
		},
		{
			sample: 'ready rows order',
			spec: 'r pairs −15 to −2',
			text: 'r is pulled in toward the round letters after it (re, ro, rc, rd, rq), closing the gap its short arm leaves.',
			title: 'An even r'
		},
		{
			sample: 'Save  24  GGUF',
			spec: 'Line 1.25 · cap-centred',
			text: 'The line box is centred on the cap height, not the x-height, so an icon centred in a row lines up with the capitals and figures beside it.',
			title: 'Centred on the caps'
		},
		{
			sample: 'typing a query',
			spec: 'Descender −' + Math.abs(metrics.dsc),
			text: 'Slightly deeper descenders keep g, p, q and y open at the tight 1.25 line, and leave room under a row of text.',
			title: 'Room below the line'
		},
		{
			sample: 'attest often',
			spec: 'Short t crossbar',
			text: 'A narrow t with short arms keeps tt, ft and st from knotting together in running text. Llama Mono keeps the long crossbar its cell needs.',
			title: 'A quieter t'
		}
	];

	const display: Highlight[] = [
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

<Section columns label="The sans of the Llama family">
	<p>
		A neo-grotesk with an even stroke, round bowls and open, angle-cut terminals, in one variable
		font: every weight from {axis.min} to {axis.max}, {glyphCount} glyphs, and the Llama icon built in
		at U+E000.
	</p>

	<p>
		It has an optical size axis from 16 to 40, which the browser sets from the font-size. Its two
		ends have names: UI at 16, for interfaces and running text, and Display at 40, for headlines.
		Sizes in between blend.
	</p>
</Section>

<Section label="Clear glass">
	<p>
		Beatrice Warde compared good type to a crystal goblet: clear enough that you see the wine, not
		the glass. Llama Sans aims for the same in an interface. Even strokes, calm round bowls and an
		even rhythm keep it out of the way of labels, messages and long reading alike, so it can carry a
		whole product without asking for attention.
	</p>

	<p>
		It isn't featureless, though. A few angled cuts give it just enough character to lead the eye
		along the line, and they come forward only when you look for them.
	</p>

	<Closeups />
</Section>

<Section label="Optical size">
	<OpticalSizes />
</Section>

<Part
	intro="The design system's text, from buttons, menus and labels to chat messages and long reading. The browser picks it on its own for text up to 16 px, spaced and kerned for those sizes."
	name="UI"
	opsz={16}
	specs="Tracking +3 · kerning matched to SF Pro · line 1.25, cap-centred"
	statement="For the interface and the page around it."
>
	<Section label="Highlights">
		<Highlights face="var(--ui)" items={ui} />
	</Section>

	<Section label="In controls">
		<UISample />
	</Section>

	<Section label="With Lucide icons">
		<IconsSample />
	</Section>
</Part>

<Part
	intro="Headings, titles, posters and anything set large. There's no separate font to pick: from 16 to 40 px the spacing moves on its own from UI's to Display's, and from 40 px up it's all Display."
	name="Display"
	opsz={40}
	specs="Tracking −22 · kerning 115% · line-height 1.15"
	statement="Quiet letters for loud ideas."
>
	<Section label="Highlights">
		<Highlights face="var(--headline)" items={display} large />
	</Section>

	<Section label="In headings">
		<DisplaySample />
	</Section>

	<Posters />
</Part>

<Section label="Numbers">
	<Metrics />
</Section>

<CharGrid />

<WeightLines />

<SansNotes />
