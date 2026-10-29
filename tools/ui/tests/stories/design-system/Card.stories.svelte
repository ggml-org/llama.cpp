<script lang="ts" module>
	import { defineMeta } from '@storybook/addon-svelte-csf';
	import { Button } from '$lib/components/ui/button';
	import * as Card from '$lib/components/ui/card';

	const { Story } = defineMeta({
		component: Card.Root,
		parameters: {
			layout: 'centered'
		},
		title: 'Design system/Card'
	});
</script>

<Story asChild name="Default">
	<Card.Root class="w-96">
		<Card.Header>
			<Card.Title>Qwen3.8-27B</Card.Title>

			<Card.Description>131k context, vision and tool use</Card.Description>
		</Card.Header>

		<Card.Content class="text-sm text-muted-foreground">
			Loaded on the server and ready to answer.
		</Card.Content>

		<Card.Footer class="justify-end gap-2">
			<Button size="sm" variant="ghost">Unload</Button>

			<Button size="sm">Start chat</Button>
		</Card.Footer>
	</Card.Root>
</Story>
