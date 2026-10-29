<script lang="ts" module>
	import StateGallery from '../../helpers/StateGallery.svelte';
	import StateGalleryItem from '../../helpers/StateGalleryItem.svelte';
	import { defineMeta } from '@storybook/addon-svelte-csf';

	// the matte utilities from src/app.css, each on a neutral sample
	const { Story } = defineMeta({
		parameters: {
			layout: 'centered'
		},
		title: 'Design system/Foundations/Surfaces'
	});
</script>

<Story asChild name="Default">
	<StateGallery columns={3}>
		<StateGalleryItem label="matte-glass" note="chat form, cards">
			<div class="matte-glass h-28 rounded-3xl"></div>
		</StateGalleryItem>

		<StateGalleryItem label="matte-glass" note="focus within">
			<div class="matte-glass flex h-28 items-center rounded-3xl px-4">
				<input
					class="w-full bg-transparent text-sm outline-none placeholder:engraved"
					placeholder="Click to see the focus edge"
				/>
			</div>
		</StateGalleryItem>

		<StateGalleryItem label="matte-float" note="menus, popovers, dialogs">
			<div class="matte-float h-28 rounded-xl border"></div>
		</StateGalleryItem>

		<StateGalleryItem label="matte-key" note="buttons keep their own fill">
			<div class="flex h-28 items-center justify-center gap-3">
				<div class="matte-key size-10 rounded-full bg-background"></div>

				<div class="matte-key h-9 w-24 rounded-md bg-secondary"></div>
			</div>
		</StateGalleryItem>

		<StateGalleryItem label="matte-well" note="inputs, switches">
			<div class="flex h-28 items-center">
				<div class="matte-well h-9 w-full rounded-md border border-input bg-background"></div>
			</div>
		</StateGalleryItem>

		<StateGalleryItem label="matte-chip" note="badges">
			<div class="flex h-28 items-center gap-2">
				<span class="matte-chip rounded-md bg-secondary px-2 py-0.5 text-xs font-medium">chip</span>

				<span class="matte-chip rounded-md bg-primary px-2 py-0.5 text-xs text-primary-foreground"
					>chip</span
				>
			</div>
		</StateGalleryItem>

		<StateGalleryItem label="matte-sheen" note="user message bubble">
			<div class="matte-sheen flex h-28 items-center rounded-2xl bg-primary/5 px-4 text-sm">
				A user message
			</div>
		</StateGalleryItem>

		<StateGalleryItem label="engraved" note="placeholder text">
			<div class="flex h-28 items-center text-sm text-muted-foreground engraved">
				Ask anything...
			</div>
		</StateGalleryItem>

		<StateGalleryItem label="status-lamp" note="red, yellow, green">
			<div class="flex h-28 items-center gap-4">
				<span class="status-lamp orb-danger"></span>

				<span class="status-lamp orb-warning"></span>

				<span class="status-lamp orb-success"></span>
			</div>
		</StateGalleryItem>

		<StateGalleryItem label="matte-orb" note="recording button">
			<div class="flex h-28 items-center gap-4">
				<span class="matte-orb orb-danger size-8 rounded-full"></span>

				<span class="matte-orb orb-warning size-8 rounded-full"></span>

				<span class="matte-orb orb-success size-8 rounded-full"></span>
			</div>
		</StateGalleryItem>
	</StateGallery>
</Story>
