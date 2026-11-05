<script lang="ts" module>
	import { seedChatStory } from '../../chat/messages/fixtures';
	import StateGallery from '../../helpers/StateGallery.svelte';
	import StateGalleryItem from '../../helpers/StateGalleryItem.svelte';
	import Boards from './palette/Boards.svelte';
	import PaletteRow from './palette/PaletteRow.svelte';
	import { basePalette, haasPalette } from './palette/palettes';
	import StatusProposal from './palette/StatusProposal.svelte';
	import { CircleAlert, Info } from '@lucide/svelte';
	import { defineMeta } from '@storybook/addon-svelte-csf';
	import * as Alert from '$lib/components/ui/alert';
	import { Badge } from '$lib/components/ui/badge';

	// Proposal: the palette. The UI stays neutral and uses colour only for status; communications start from the
	// neutral base, and the Haas palette adds the colour on top of it. Delete once the colours are in the tokens
	// (see DESIGN.md, "Proposing changes").
	const { Story } = defineMeta({
		parameters: {
			controls: { disable: true },
			layout: 'fullscreen'
		},
		title: 'Design System/Proposals/Palette'
	});
</script>

<Story asChild name="UI status">
	<StateGallery columns={3}>
		<StateGalleryItem label="Current" note="Alert: default and destructive only">
			<div class="grid gap-2">
				<Alert.Root>
					<Info />

					<Alert.Title>Update available</Alert.Title>

					<Alert.Description>llama.cpp b9726 adds faster prompt processing.</Alert.Description>
				</Alert.Root>

				<Alert.Root variant="destructive">
					<CircleAlert />

					<Alert.Title>Download failed</Alert.Title>

					<Alert.Description>The connection dropped at 64%. Try again.</Alert.Description>
				</Alert.Root>
			</div>
		</StateGalleryItem>

		<StateGalleryItem label="Proposed" note="Success, warning, error and info, on a soft surface">
			<StatusProposal part="alerts" />
		</StateGalleryItem>

		<StateGalleryItem label="Saturated" note="The same hues with more colour">
			<StatusProposal part="alerts" saturated />
		</StateGalleryItem>

		<StateGalleryItem label="Current" note="Badge: no success, warning or info">
			<div class="flex flex-wrap gap-2">
				<Badge>Default</Badge>

				<Badge variant="secondary">Secondary</Badge>

				<Badge variant="outline">Outline</Badge>

				<Badge variant="destructive">Failed</Badge>
			</div>
		</StateGalleryItem>

		<StateGalleryItem label="Proposed" note="Status badges with a dot">
			<StatusProposal part="badges" />
		</StateGalleryItem>

		<StateGalleryItem label="Saturated" note="The same hues with more colour">
			<StatusProposal part="badges" saturated />
		</StateGalleryItem>

		<StateGalleryItem label="Current" note="Status dots: success, warning, danger">
			<div class="flex flex-wrap items-center gap-4 text-[13px]">
				{#each [['success', 'Loaded'], ['warning', 'Too large'], ['danger', 'Failed']] as [k, l] (k)}
					<span class="inline-flex items-center gap-1.5">
						<i
							style:background="var(--status-{k})"
							style:box-shadow="0 0 0 3px var(--status-{k}-rim)"
							class="size-2 rounded-full"
						></i>{l}
					</span>
				{/each}
			</div>
		</StateGalleryItem>

		<StateGalleryItem label="Proposed" note="One hue per meaning, a soft halo, an info dot">
			<StatusProposal part="dots" />
		</StateGalleryItem>

		<StateGalleryItem label="Saturated" note="The same hues with more colour">
			<StatusProposal part="dots" saturated />
		</StateGalleryItem>

		<StateGalleryItem label="Current" note="Hint text: danger and success only">
			<div class="grid gap-1.5 text-[13px]">
				<p class="m-0 text-fg-success">Port 9931 is free.</p>

				<p class="m-0 text-fg-danger">Enter a number between 1 and 128.</p>
			</div>
		</StateGalleryItem>

		<StateGalleryItem label="Proposed" note="All four, at text contrast">
			<StatusProposal part="text" />
		</StateGalleryItem>

		<StateGalleryItem label="Saturated" note="The same hues with more colour">
			<StatusProposal part="text" saturated />
		</StateGalleryItem>
	</StateGallery>
</Story>

<Story asChild beforeEach={() => seedChatStory()} name="Communications base">
	<StateGallery columns={1}>
		<StateGalleryItem label={basePalette.name} note={basePalette.note}>
			<PaletteRow palette={basePalette} />
		</StateGalleryItem>
	</StateGallery>
</Story>

<Story asChild beforeEach={() => seedChatStory()} name="Specimen boards">
	<StateGallery columns={1}>
		<StateGalleryItem label="Key visuals" note={haasPalette.note}>
			<PaletteRow palette={haasPalette} />
		</StateGalleryItem>

		<StateGalleryItem
			label="Specimen boards"
			note="After the classic Swiss specimen boards, in the same palette"
		>
			<Boards />
		</StateGalleryItem>
	</StateGallery>
</Story>
