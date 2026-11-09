<script lang="ts" module>
	import glyphs from './llama-mono/data/glyphs.json';
	import Diff from './llama-mono/Diff.svelte';
	import KeyVisuals from './llama-mono/KeyVisuals.svelte';
	import Logs from './llama-mono/Logs.svelte';
	import MonoInterface from './llama-mono/MonoInterface.svelte';
	import MonoPage from './llama-mono/MonoPage.svelte';
	import Terminal from './llama-mono/Terminal.svelte';
	import GlyphGrid from './llama-sans/lib/components/GlyphGrid.svelte';
	import type { Specimen } from './llama-sans/lib/specimen';
	import OverviewFrame from './llama-sans/overview/OverviewFrame.svelte';
	import SpecimenFrame from './llama-sans/SpecimenFrame.svelte';
	import TypographyOverview from './typography/TypographyOverview.svelte';
	import { defineMeta } from '@storybook/addon-svelte-csf';

	// Llama Mono: its overview, then in use: the terminal, the user interface beside Llama Sans, logs and tables, diffs
	// and its key visuals; then every glyph, at Llama Sans's cell size (the llama.cpp lockup is Brand/Logo/Lockup)
	const { Story } = defineMeta({
		parameters: {
			controls: { disable: true },
			layout: 'fullscreen'
		},
		title: 'Brand/Typography/Llama Mono'
	});
</script>

<Story asChild name="Overview">
	<OverviewFrame>
		<TypographyOverview variant="mono" />
	</OverviewFrame>
</Story>

<Story asChild name="Terminal">
	<OverviewFrame>
		<MonoPage
			intro="Where Llama Mono lives first: llama-server and llama-cli in a dark and a light terminal, with the Llama icon in the prompt."
			kicker="Llama Mono · in the terminal"
			title="Terminal"
		>
			<Terminal />
		</MonoPage>
	</OverviewFrame>
</Story>

<Story asChild name="User Interface">
	<MonoInterface />
</Story>

<Story asChild name="Logs">
	<OverviewFrame>
		<MonoPage
			intro="Llama Mono where text has to line up without a grid: a server's request log and a benchmark table."
			kicker="Llama Mono · logs and tables"
			title="Logs"
		>
			<Logs />
		</MonoPage>
	</OverviewFrame>
</Story>

<Story asChild name="Diff">
	<OverviewFrame>
		<MonoPage
			intro="Llama Mono in a diff: a change to llama.cpp's defaults, unified on a light and a dark ground, then split, the old file beside the new."
			kicker="Llama Mono · in review"
			title="Diff"
		>
			<Diff />
		</MonoPage>
	</OverviewFrame>
</Story>

<Story asChild name="Glyphs">
	<SpecimenFrame variant="UI">
		<GlyphGrid comparing={false} grid={glyphs as unknown as Specimen['grid']} weight={400} />
	</SpecimenFrame>
</Story>

<Story asChild name="Key visuals">
	<OverviewFrame>
		<MonoPage
			intro="Four neutral posters, each set in the font's own cells: a terminal, the look-alikes kept apart, a generation speed in tabular figures and the first bytes of a GGUF file."
			kicker="Llama Mono · for communications"
			title="Key visuals"
		>
			<KeyVisuals />
		</MonoPage>
	</OverviewFrame>
</Story>
