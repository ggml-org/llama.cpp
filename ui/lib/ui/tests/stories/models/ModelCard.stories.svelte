<script lang="ts" module>
	import StateGallery from '../helpers/StateGallery.svelte';
	import StateGalleryItem from '../helpers/StateGalleryItem.svelte';
	import { defineMeta } from '@storybook/addon-svelte-csf';
	import { Database, Download, ScrollText } from '@lucide/svelte';
	import { ModelCapabilityIcons, ModelCard, ModelId, ModelOrgAvatar } from '$lib/components/models';

	// the surface every model card shares, Llama Chat's Discover and models page among them: the avatar and name are the
	// card's one button (or link), capabilities and a description in the body, facts at the foot
	const { Story } = defineMeta({
		component: ModelCard,
		parameters: {
			layout: 'centered'
		},
		title: 'UI/Blocks/Models/ModelCard'
	});
</script>

{#snippet card(selected = false, dimmed = false)}
	<ModelCard class="w-72" {dimmed} onSelect={() => {}} {selected} tooltip="ggml-org/gpt-oss-20b-GGUF">
		{#snippet avatar()}
			<ModelOrgAvatar org="openai" quantOrg="ggml-org" />
		{/snippet}

		{#snippet title()}
			<ModelId hideOrgName name="gpt-oss-20b-GGUF" params="20B" stackId />
		{/snippet}

		<ModelCapabilityIcons asBadges supportsThinking supportsToolUse />

		<p class="line-clamp-2 text-xs text-muted-foreground">
			OpenAI's open-weight model for local reasoning and agents, with tool use and adjustable reasoning effort.
		</p>

		{#snippet footer()}
			<span class="flex items-center gap-3 whitespace-nowrap">
				<span class="inline-flex items-center gap-1"><Database class="size-icon-xs" />12.1 GB</span>
				<span class="inline-flex items-center gap-1"><ScrollText class="size-icon-xs" />131.07K</span>
			</span>
		{/snippet}

		{#snippet footerEnd()}
			<span class="inline-flex items-center gap-1 text-muted-foreground"><Download class="size-icon-xs" />1.2M</span>
		{/snippet}
	</ModelCard>
{/snippet}

<Story asChild name="Default">
	<StateGallery columns={3} minItemWidth="18rem">
		<StateGalleryItem label="default">{@render card()}</StateGalleryItem>
		<StateGalleryItem label="selected" note="its details are open">{@render card(true)}</StateGalleryItem>
		<StateGalleryItem label="dimmed" note="a hidden model">{@render card(false, true)}</StateGalleryItem>
	</StateGallery>
</Story>
