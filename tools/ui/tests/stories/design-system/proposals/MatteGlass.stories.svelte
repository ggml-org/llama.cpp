<script lang="ts" module>
	import StateGallery from '../../helpers/StateGallery.svelte';
	import StateGalleryItem from '../../helpers/StateGalleryItem.svelte';
	import MatteSet from './matte/MatteSet.svelte';
	import { Send } from '@lucide/svelte';
	import { defineMeta } from '@storybook/addon-svelte-csf';
	import { Badge } from '$lib/components/ui/badge';
	import { Button } from '$lib/components/ui/button';

	// Proposal: a matte-glass finish (gradients between near shades of one colour, a soft lighter edge, pressed-in
	// glyphs). Delete once chosen (see DESIGN.md, "Proposing changes").
	const { Story } = defineMeta({
		parameters: {
			controls: { disable: true },
			layout: 'fullscreen'
		},
		title: 'Design System/Proposals/Matte glass'
	});
</script>

<Story asChild name="Finish">
	<StateGallery columns={2}>
		<StateGalleryItem label="Current" note="Flat lamps; there is no tray today">
			<div class="flex items-center gap-4 p-4">
				<span class="status-lamp orb-danger size-3 rounded-full"></span>

				<span class="status-lamp orb-warning size-3 rounded-full"></span>

				<span class="status-lamp orb-success size-3 rounded-full"></span>
			</div>
		</StateGalleryItem>

		<StateGalleryItem label="Proposed" note="A quiet tray with near-flat coloured keys">
			<div class="p-4"><MatteSet part="tray" /></div>
		</StateGalleryItem>

		<StateGalleryItem label="Current" note="The solid destructive badge">
			<div class="flex items-center gap-3 p-4">
				<Badge variant="destructive">6</Badge>

				<Badge variant="destructive">12</Badge>
			</div>
		</StateGalleryItem>

		<StateGalleryItem
			label="Proposed"
			note="A thin collar, a near-flat fill, a lightly raised count"
		>
			<div class="p-4"><MatteSet part="count" /></div>
		</StateGalleryItem>

		<StateGalleryItem label="Current" note="Flat soft pills with a dot">
			<div class="flex flex-wrap items-center gap-2 p-4">
				<Badge variant="success">Ready</Badge>

				<Badge variant="warning">Loading</Badge>

				<Badge variant="danger">Failed</Badge>

				<Badge variant="info">Info</Badge>

				<Badge variant="secondary">Q4_K_M</Badge>
			</div>
		</StateGalleryItem>

		<StateGalleryItem label="Proposed" note="A tint a shade lighter at the top, a hairline edge">
			<div class="p-4"><MatteSet part="badges" /></div>
		</StateGalleryItem>

		<StateGalleryItem label="Current" note="The buttons today">
			<div class="flex flex-wrap items-center gap-3 p-4">
				<Button>Load model</Button>

				<Button variant="secondary">Cancel</Button>

				<Button class="rounded-full" size="icon"><Send /></Button>
			</div>
		</StateGalleryItem>

		<StateGalleryItem
			label="Proposed"
			note="A shade lighter at the top, a faint lit top edge, no glow"
		>
			<div class="p-4"><MatteSet part="buttons" /></div>
		</StateGalleryItem>
	</StateGallery>
</Story>
