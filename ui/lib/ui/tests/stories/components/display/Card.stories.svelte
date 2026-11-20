<script lang="ts" module>
	import { Button } from '@ggml-org/llama-ui/ui/button';
	import * as Card from '@ggml-org/llama-ui/ui/card';
	import { defineMeta } from '@storybook/addon-svelte-csf';

	const { Story } = defineMeta({
		component: Card.Root,
		parameters: {
			layout: 'centered'
		},
		title: 'Llama UI/Components/Display/Card'
	});
</script>

<Story asChild name="Default">
	<Card.Root class="w-96">
		<Card.Header>
			<Card.Title>Qwen3.8-27B</Card.Title>

			<Card.Description>131k context, vision and tool use</Card.Description>
		</Card.Header>

		<Card.Content class="text-body-sm text-muted-foreground">
			Loaded on the server and ready to answer.
		</Card.Content>

		<Card.Footer class="justify-end gap-2">
			<Button size="sm" variant="ghost">Unload</Button>

			<Button size="sm">Start chat</Button>
		</Card.Footer>
	</Card.Root>
</Story>
