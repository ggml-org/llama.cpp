<script lang="ts" module>
	import { ServerRole } from '$lib/enums';
	import { ModelsService } from '$lib/services';
	import { modelsStore, serverStore } from '$lib/stores';
	import type { ModelOption } from '$lib/types/models';

	const model = (id: string, name: string, orgName: string, params: string): ModelOption => ({
		capabilities: [],
		id,
		model: id,
		name,
		parsedId: {
			activatedParams: null,
			modelName: name,
			orgName,
			params,
			quantization: 'Q4_K_M',
			raw: id,
			sidecar: null,
			tags: []
		}
	});

	const MODELS = [
		model('ggml-org/Qwen3-8B-GGUF:Q4_K_M', 'Qwen3-8B', 'ggml-org', '8B'),
		model('unsloth/Llama-3.2-3B-GGUF:Q4_K_M', 'Llama-3.2-3B', 'unsloth', '3B')
	];

	// a router with models to pick from, the first one selected: what the chat form shows
	// with a server behind it. The selector fetches its list on mount, so it is served the same
	serverStore.role = ServerRole.ROUTER;
	ModelsService.list = async () => ({ data: MODELS.map(({ id }) => ({ id })) }) as never;
	modelsStore.models = MODELS;
	modelsStore.selectedModelId = MODELS[0].id;
</script>

<script lang="ts">
	import ChatForm from '$lib/components/app/chat/ChatForm/ChatForm.svelte';
	import { untrack } from 'svelte';

	interface Props {
		initial?: string;
		/** The single row's corner radius, any CSS length. */
		radius?: string;
		/** The chat form without the single row, as it was before. */
		today?: boolean;
		width?: string;
	}

	let { initial = '', radius, today = false, width = '52rem' }: Props = $props();

	let value = $state(untrack(() => initial));
</script>

<div style:--chat-form-single-row-radius={radius} style:width>
	<ChatForm bind:value singleRow={!today} />
</div>
