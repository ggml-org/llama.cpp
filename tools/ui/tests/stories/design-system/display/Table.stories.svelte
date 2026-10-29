<script lang="ts" module>
	import { defineMeta } from '@storybook/addon-svelte-csf';
	import * as Table from '$lib/components/ui/table';

	const { Story } = defineMeta({
		component: Table.Root,
		parameters: {
			layout: 'centered'
		},
		title: 'Design system/Display/Table'
	});

	const rows = [
		{ context: '131k', model: 'Qwen3.8-27B', quant: 'Q4_K_M', size: '16.4 GB' },
		{ context: '131k', model: 'Qwen3.8-27B', quant: 'Q8_0', size: '28.7 GB' },
		{ context: '8k', model: 'Llama-3.2-3B', quant: 'Q4_K_M', size: '2.0 GB' }
	];
</script>

<Story asChild name="Default">
	<div class="w-[36rem]">
		<Table.Root>
			<Table.Caption>Models on this server</Table.Caption>

			<Table.Header>
				<Table.Row>
					<Table.Head>Model</Table.Head>

					<Table.Head>Quant</Table.Head>

					<Table.Head>Context</Table.Head>

					<Table.Head class="text-right">Size</Table.Head>
				</Table.Row>
			</Table.Header>

			<Table.Body>
				{#each rows as row (row.model + row.quant)}
					<Table.Row>
						<Table.Cell class="font-medium">{row.model}</Table.Cell>

						<Table.Cell>{row.quant}</Table.Cell>

						<Table.Cell>{row.context}</Table.Cell>

						<Table.Cell class="text-right">{row.size}</Table.Cell>
					</Table.Row>
				{/each}
			</Table.Body>

			<Table.Footer>
				<Table.Row>
					<Table.Cell colspan={3}>Total</Table.Cell>

					<Table.Cell class="text-right">47.1 GB</Table.Cell>
				</Table.Row>
			</Table.Footer>
		</Table.Root>
	</div>
</Story>
