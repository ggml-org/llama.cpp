<!--
	Llama Sans in the user interface, built from the design system's own components (tabs, fields, toggle groups,
	switches, badges, buttons, cards and a table) with llama.cpp's content, at their real sizes. Nothing pins the
	optical size: text from 12 to 16 px lands on the UI end of the axis by itself, and the card titles blend a little
	toward Display.
-->
<script lang="ts">
	import { Download, Paperclip, Plus, Power, RotateCcw } from '@lucide/svelte';
	import { Badge } from '$lib/components/ui/badge';
	import { Button } from '$lib/components/ui/button';
	import * as Card from '$lib/components/ui/card';
	import { Input } from '$lib/components/ui/input';
	import { Label } from '$lib/components/ui/label';
	import { Switch } from '$lib/components/ui/switch';
	import * as Table from '$lib/components/ui/table';
	import * as Tabs from '$lib/components/ui/tabs';
	import * as ToggleGroup from '$lib/components/ui/toggle-group';

	let context = $state('32768');
	let cache = $state('q8_0');

	const server = $state([
		{ hint: 'Faster attention on supported GPUs', id: 'flash', label: 'Flash attention', on: true },
		{
			hint: 'Serve several requests in one batch',
			id: 'batch',
			label: 'Continuous batching',
			on: true
		},
		{
			hint: 'Expose Prometheus metrics at /metrics',
			id: 'metrics',
			label: 'Metrics endpoint',
			on: false
		}
	]);

	const models = [
		{ context: '131k', name: 'Qwen3.8-27B', quant: 'Q4_K_M', size: '16.4 GB', speed: '42.7' },
		{ context: '128k', name: 'gemma-3-12b-it', quant: 'Q8_0', size: '12.5 GB', speed: '38.1' },
		{ context: '32k', name: 'SmolLM3-3B', quant: 'F16', size: '6.2 GB', speed: '96.4' }
	];
</script>

<div class="bg-background px-4 py-10 text-foreground sm:px-8">
	<div class="mx-auto grid max-w-5xl gap-8">
		<header class="grid gap-2">
			<p class="font-mono text-caption tracking-widest text-muted-foreground uppercase">
				Llama Sans · opsz 16, from the size
			</p>

			<h1 class="text-4xl font-bold tracking-tight">User Interface</h1>

			<p class="max-w-[62ch] text-body text-muted-foreground">
				The design system's own components, at their real sizes. Nothing here pins the optical size:
				text from 12 to 16 px lands on the UI end of the axis by itself, spaced and kerned for
				controls, with its line box centred on the cap height so labels sit in the middle of buttons
				and badges.
			</p>
		</header>

		<div class="flex flex-wrap items-center gap-2">
			<Button><Plus /> New chat</Button>

			<Button variant="secondary"><Paperclip /> Attach</Button>

			<Button variant="outline"><Download /> Export</Button>

			<Button variant="ghost">Cancel</Button>

			<Button variant="destructive"><Power /> Unload</Button>

			<span aria-hidden="true" class="mx-2 h-5 w-px bg-border"></span>

			<Badge variant="success">Ready</Badge>

			<Badge variant="info">3 slots</Badge>

			<Badge variant="warning">Context 92%</Badge>

			<Badge variant="secondary">v0.6</Badge>
		</div>

		<div class="grid gap-4 md:grid-cols-2">
			<Card.Root>
				<Card.Header>
					<Card.Title>Model settings</Card.Title>

					<Card.Description>Qwen3.8-27B · Q4_K_M</Card.Description>
				</Card.Header>

				<Card.Content class="grid gap-5">
					<Tabs.Root value="general">
						<Tabs.List>
							<Tabs.Trigger value="general">General</Tabs.Trigger>

							<Tabs.Trigger value="sampling">Sampling</Tabs.Trigger>

							<Tabs.Trigger value="server">Server</Tabs.Trigger>
						</Tabs.List>
					</Tabs.Root>

					<div class="grid gap-2">
						<Label for="ctx-size">Context size</Label>

						<Input bind:value={context} id="ctx-size" inputmode="numeric" />

						<p class="text-caption text-muted-foreground">
							Tokens the model keeps in view. Up to 131,072 for this model.
						</p>
					</div>

					<div class="grid gap-2">
						<Label>KV cache</Label>

						<ToggleGroup.Root bind:value={cache} type="single">
							<ToggleGroup.Item value="f16">F16</ToggleGroup.Item>

							<ToggleGroup.Item value="q8_0">Q8_0</ToggleGroup.Item>

							<ToggleGroup.Item value="q4_0">Q4_0</ToggleGroup.Item>
						</ToggleGroup.Root>
					</div>
				</Card.Content>

				<Card.Footer class="justify-end gap-2">
					<Button variant="ghost"><RotateCcw /> Reset</Button>

					<Button>Save changes</Button>
				</Card.Footer>
			</Card.Root>

			<Card.Root>
				<Card.Header>
					<Card.Title>Server</Card.Title>

					<Card.Description>llama-server on 127.0.0.1:9931</Card.Description>
				</Card.Header>

				<Card.Content class="grid gap-4">
					{#each server as s (s.id)}
						<div class="flex items-start justify-between gap-4">
							<div class="grid gap-0.5">
								<Label for="srv-{s.id}">{s.label}</Label>

								<p class="text-caption text-muted-foreground">{s.hint}</p>
							</div>

							<Switch bind:checked={s.on} id="srv-{s.id}" />
						</div>
					{/each}
				</Card.Content>
			</Card.Root>

			<Card.Root class="md:col-span-2">
				<Card.Header>
					<Card.Title>Loaded models</Card.Title>

					<Card.Description>Generation speed on this machine, in tokens per second</Card.Description
					>
				</Card.Header>

				<Card.Content>
					<Table.Root>
						<Table.Header>
							<Table.Row>
								<Table.Head>Model</Table.Head>

								<Table.Head>Quant</Table.Head>

								<Table.Head class="text-right">Context</Table.Head>

								<Table.Head class="text-right">Size</Table.Head>

								<Table.Head class="text-right">tok/s</Table.Head>
							</Table.Row>
						</Table.Header>

						<Table.Body>
							{#each models as m (m.name)}
								<Table.Row>
									<Table.Cell class="font-medium">{m.name}</Table.Cell>

									<Table.Cell><span class="font-mono">{m.quant}</span></Table.Cell>

									<Table.Cell class="text-right tabular-nums">{m.context}</Table.Cell>

									<Table.Cell class="text-right tabular-nums">{m.size}</Table.Cell>

									<Table.Cell class="text-right tabular-nums">{m.speed}</Table.Cell>
								</Table.Row>
							{/each}
						</Table.Body>
					</Table.Root>
				</Card.Content>
			</Card.Root>
		</div>
	</div>
</div>
