<!--
	Llama Sans UI's part of the Typography overview, after its hero: what it's for, what sets it apart (highlights),
	the details up close, at work in controls and with Lucide icons, the numbers, the characters and weights set in it,
	and the shared design notes.
-->
<script lang="ts">
	import Metrics from '../llama-sans/lib/components/Metrics.svelte';
	import { specimen } from '../llama-sans/lib/specimen';
	import CharGrid from '../llama-sans/overview/CharGrid.svelte';
	import Closeups from '../llama-sans/overview/Closeups.svelte';
	import IconsSample from '../llama-sans/overview/IconsSample.svelte';
	import Section from '../llama-sans/overview/Section.svelte';
	import UISample from '../llama-sans/overview/UISample.svelte';
	import Version from '../llama-sans/overview/Version.svelte';
	import WeightLines from '../llama-sans/overview/WeightLines.svelte';
	import Highlights, { type Highlight } from './Highlights.svelte';
	import SansNotes from './SansNotes.svelte';

	const face = 'var(--ui)';

	const highlights: Highlight[] = [
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
			spec: 'Descender −' + Math.abs(specimen.metrics.dsc),
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
</script>

<Section columns label="For interfaces and running text">
	<p>
		Llama Sans UI is the design system's text face: buttons, menus, labels, chat messages and long
		reading. It shares its outlines with Display; only the spacing, kerning and line box differ,
		tuned for 12 to 18 px.
	</p>

	<p>
		{specimen.glyphCount} glyphs, every weight from {specimen.axis.min} to {specimen.axis.max} in one
		variable font, and the Llama icon built in at U+E000.
	</p>
</Section>

<Section label="Highlights">
	<Highlights {face} items={highlights} />
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

<Version
	name="UI"
	specs="Tracking +3 · kerning matched to SF Pro · line 1.25, cap-centred"
	statement="For the interface and the page around it."
>
	<UISample />
</Version>

<Section label="With Lucide icons">
	<IconsSample />
</Section>

<Section label="Numbers">
	<Metrics />
</Section>

<CharGrid {face} />

<WeightLines {face} />

<SansNotes />
