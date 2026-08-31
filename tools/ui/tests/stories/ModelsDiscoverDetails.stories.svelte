<script lang="ts" module>
	import { mockDetails, mockSiblings } from './fixtures/models-discover';
	import { defineMeta } from '@storybook/addon-svelte-csf';
	import ModelsDiscoverChatTemplateDialog from '$lib/components/app/models/discover/ModelsDiscoverChatTemplateDialog.svelte';
	import ModelsDiscoverDetails from '$lib/components/app/models/discover/ModelsDiscoverDetails.svelte';
	import ModelsDiscoverDetailsDownloadOptions from '$lib/components/app/models/discover/ModelsDiscoverDetailsDownloadOptions.svelte';
	import ModelsDiscoverDetailsHeader from '$lib/components/app/models/discover/ModelsDiscoverDetailsHeader.svelte';
	import ModelsDiscoverDetailsName from '$lib/components/app/models/discover/ModelsDiscoverDetailsName.svelte';
	import ModelsDiscoverDetailsReadme from '$lib/components/app/models/discover/ModelsDiscoverDetailsReadme.svelte';
	import TerminalCommands from '$lib/components/app/models/discover/TerminalCommands.svelte';
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
		<ModelsDiscoverDetailsHeader
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

<Story name="Name">
	<div class="space-y-4 p-4">
		<ModelsDiscoverDetailsName
			baseModels={['google/gemma-4-12b-it']}
			hasReasoning
			hasTools
			hasVision
			modelId="ggml-org/gemma-4-12b-it"
		/>

		<ModelsDiscoverDetailsName
			baseModels={[]}
			hasReasoning={false}
			hasTools={false}
			hasVision={false}
			modelId="ggml-org/Qwen3.8-27B"
		/>
	</div>
</Story>

<Story name="Download options">
	<div class="w-200 p-4">
		<ModelsDiscoverDetailsDownloadOptions
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
		<ModelsDiscoverDetailsDownloadOptions
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
		<TerminalCommands modelId="ggml-org/Qwen3.8-27B-GGUF" />
	</div>
</Story>

<Story name="Terminal commands (MTP sidecar)">
	<div class="w-200 p-4">
		<TerminalCommands modelId="ggml-org/gemma-4-12b-it-GGUF" sidecars={[ModelDraftSidecar.MTP]} />
	</div>
</Story>

<Story name="Readme">
	<div class="w-160 p-4">
		<ModelsDiscoverDetailsReadme {readme} />
	</div>
</Story>

<Story name="Chat template dialog">
	<div class="p-4">
		<ModelsDiscoverChatTemplateDialog chatTemplate={CHAT_TEMPLATE} open />
	</div>
</Story>

<Story name="Details (loading)">
	<div class="h-96 w-200 border">
		<ModelsDiscoverDetails
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
		<ModelsDiscoverDetails
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
		<ModelsDiscoverDetails
			details={mockDetails}
			{files}
			modelId={mockDetails.id ?? 'ggml-org/gemma-4-12b-it-GGUF'}
			{readme}
		/>
	</div>
</Story>
