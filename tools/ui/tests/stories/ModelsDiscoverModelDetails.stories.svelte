<script lang="ts" module>
	import { mockDetails, mockSiblings } from './fixtures/models-discover';
	import { defineMeta } from '@storybook/addon-svelte-csf';
	import ModelsDiscoverChatTemplateDialog from '$lib/components/app/models/discover/ModelsDiscoverChatTemplateDialog.svelte';
	import ModelsDiscoverModelDetails from '$lib/components/app/models/discover/ModelsDiscoverModelDetails.svelte';
	import ModelsDiscoverModelDetailsCommands from '$lib/components/app/models/discover/ModelsDiscoverModelDetailsCommands.svelte';
	import ModelsDiscoverModelDetailsDownloadOptions from '$lib/components/app/models/discover/ModelsDiscoverModelDetailsDownloadOptions.svelte';
	import ModelsDiscoverModelDetailsHeader from '$lib/components/app/models/discover/ModelsDiscoverModelDetailsHeader.svelte';
	import ModelsDiscoverModelDetailsReadme from '$lib/components/app/models/discover/ModelsDiscoverModelDetailsReadme.svelte';
	import { ModelDraftSidecar } from '$lib/enums';
	import type { HfModelSibling } from '$lib/types';

	const { Story } = defineMeta({
		tags: ['autodocs'],
		title: 'Models/Discover/Details'
	});

	const files: HfModelSibling[] = mockSiblings;

	const readme = [
		'# Gemma 4 12B IT',
		'',
		'Quantized ggml-org build of google/gemma-4-12b-it.',
		'',
		'| Quant | Size |',
		'| ----- | ---- |',
		'| Q4_K_M | 7.3 GB |',
		'| Q8_0 | 13.1 GB |'
	].join('\n');

	const CHAT_TEMPLATE = '{% for message in messages %}{ tools }{% endfor %}';
</script>

<Story name="Header">
	<div class="w-160 p-4">
		<ModelsDiscoverModelDetailsHeader
			baseModels={mockDetails.cardData?.base_model ? [String(mockDetails.cardData.base_model)] : []}
			details={mockDetails}
			hasReasoning
			hasTools
			hasVision
			licenseTag={mockDetails.tags
				?.find((t) => t.startsWith('license:'))
				?.replace('license:', '') ?? null}
			modelId={mockDetails.id ?? 'ggml-org/gemma-4-12b-it-GGUF'}
		/>
	</div>
</Story>

<Story name="Download options">
	<div class="w-200 p-4">
		<ModelsDiscoverModelDetailsDownloadOptions
			bitDepthRows={[
				{ bitDepth: 4, files: files.filter((f) => f.path.includes('Q4_K_M')) },
				{ bitDepth: 8, files: files.filter((f) => f.path.includes('Q8_0')) },
				{ bitDepth: 16, files: files.filter((f) => f.path.includes('BF16')) }
			]}
			modelId="ggml-org/gemma-4-12b-it-GGUF"
		/>
	</div>
</Story>

<Story name="Download options (unknown device)">
	<div class="w-200 p-4">
		<ModelsDiscoverModelDetailsDownloadOptions
			bitDepthRows={[
				{ bitDepth: 4, files: files.filter((f) => f.path.includes('Q4_K_M')) },
				{ bitDepth: 8, files: files.filter((f) => f.path.includes('Q8_0')) }
			]}
			modelId="ggml-org/gemma-4-12b-it-GGUF"
		/>
	</div>
</Story>

<Story name="Terminal commands (no sidecars)">
	<div class="w-200 p-4">
		<ModelsDiscoverModelDetailsCommands modelId="ggml-org/Qwen3.8-27B-GGUF" />
	</div>
</Story>

<Story name="Terminal commands (MTP sidecar)">
	<div class="w-200 p-4">
		<ModelsDiscoverModelDetailsCommands
			modelId="ggml-org/gemma-4-12b-it-GGUF"
			sidecars={[ModelDraftSidecar.MTP]}
		/>
	</div>
</Story>

<Story name="Readme">
	<div class="w-160 p-4">
		<ModelsDiscoverModelDetailsReadme {readme} />
	</div>
</Story>

<Story name="Chat template dialog">
	<div class="p-4">
		<ModelsDiscoverChatTemplateDialog chatTemplate={CHAT_TEMPLATE} open />
	</div>
</Story>

<Story name="Details (loading)">
	<div class="h-96 w-200 border">
		<ModelsDiscoverModelDetails
			details={null}
			files={[]}
			loading
			modelId="ggml-org/gemma-4-12b-it-GGUF"
			readme={null}
		/>
	</div>
</Story>

<Story name="Details (error)">
	<div class="h-96 w-200 border">
		<ModelsDiscoverModelDetails
			details={null}
			error="Model not found"
			files={[]}
			modelId="ggml-org/does-not-exist-GGUF"
			readme={null}
		/>
	</div>
</Story>

<Story name="Details (loaded)">
	<div class="h-96 w-200 overflow-y-auto border">
		<ModelsDiscoverModelDetails
			details={mockDetails}
			{files}
			modelId={mockDetails.id ?? 'ggml-org/gemma-4-12b-it-GGUF'}
			{readme}
		/>
	</div>
</Story>
