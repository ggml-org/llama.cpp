<script lang="ts" module>
	import StateGallery from '../helpers/StateGallery.svelte';
	import StateGalleryItem from '../helpers/StateGalleryItem.svelte';
	import { defineMeta } from '@storybook/addon-svelte-csf';
	import { ModelOrgAvatar } from '$lib/components/models';

	// the maker's Hugging Face avatar, with the quantizer's in the corner when someone else published the GGUF; an
	// org with no avatar falls back to its initial on a hue of its own
	const { Story } = defineMeta({
		component: ModelOrgAvatar,
		parameters: {
			layout: 'centered'
		},
		title: 'UI/Blocks/Models/ModelOrgAvatar'
	});
</script>

<Story asChild name="Default">
	<StateGallery columns={4} minItemWidth="8rem">
		<StateGalleryItem label="maker">
			<ModelOrgAvatar org="google" />
		</StateGalleryItem>

		<StateGalleryItem label="maker and quantizer">
			<ModelOrgAvatar org="Qwen" quantOrg="ggml-org" />
		</StateGalleryItem>

		<StateGalleryItem label="inverted in dark" note="a dark mark on transparent">
			<ModelOrgAvatar org="openai" quantOrg="ggml-org" />
		</StateGalleryItem>

		<StateGalleryItem label="no avatar" note="the monogram">
			<ModelOrgAvatar org="no-such-org-on-the-hub" />
		</StateGalleryItem>

		<StateGalleryItem label="small" note="in a list row">
			<ModelOrgAvatar org="LiquidAI" quantOrg="ggml-org" quantSize="size-3" size="size-5" />
		</StateGalleryItem>

		<StateGalleryItem label="large" note="in a model's header">
			<ModelOrgAvatar
				org="mistralai"
				quantOrg="ggml-org"
				quantPositionClass="-bottom-1.5 -right-1.5"
				quantSize="size-6"
				size="size-12"
			/>
		</StateGalleryItem>
	</StateGallery>
</Story>
