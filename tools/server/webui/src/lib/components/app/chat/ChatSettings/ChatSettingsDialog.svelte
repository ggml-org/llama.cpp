<script lang="ts">
	import { Settings, Filter, Hand, MessageSquare, Plus, Beaker } from '@lucide/svelte';
	import { ChatSettingsFooter, ChatSettingsSection } from '$lib/components/app';
	import { Checkbox } from '$lib/components/ui/checkbox';
	import * as Dialog from '$lib/components/ui/dialog';
	import { Input } from '$lib/components/ui/input';
	import { ScrollArea } from '$lib/components/ui/scroll-area';
	import { Textarea } from '$lib/components/ui/textarea';
	import { SETTING_CONFIG_DEFAULT } from '$lib/constants/settings-config';
	import {
		config,
		updateMultipleConfig,
		resetConfig
	} from '$lib/stores/settings.svelte';

	interface Props {
		onOpenChange?: (open: boolean) => void;
		open?: boolean;
	}

	let { onOpenChange, open = false }: Props = $props();

	let localConfig: SettingsConfigType = $state({ ...config() });

	$effect(() => {
		if (open) {
			localConfig = { ...config() };
		}
	});

	const defaultConfig = SETTING_CONFIG_DEFAULT;

	function handleSave() {
		updateMultipleConfig(localConfig);

		onOpenChange?.(false);
	}

	function handleReset() {
		resetConfig();

		localConfig = { ...SETTING_CONFIG_DEFAULT };
	}

	function handleClose() {
		onOpenChange?.(false);
	}

	let activeSection = $state('General');

	const settingSections: Array<{
		title: string;
		icon: any;
		fields: SettingsFieldConfig[];
	}> = [
		{
			title: 'General',
			icon: Settings,
			fields: [
				{ key: 'apiKey', label: 'API Key', type: 'input' },
				{
					key: 'systemMessage',
					label: 'System Message (will be disabled if left empty)',
					type: 'textarea'
				},
				{ key: 'temperature', label: 'Temperature', type: 'input' },
				{ key: 'top_k', label: 'Top K', type: 'input' },
				{ key: 'top_p', label: 'Top P', type: 'input' },
				{ key: 'min_p', label: 'Min P', type: 'input' },
				{ key: 'max_tokens', label: 'Max Tokens', type: 'input' },
				{ key: 'pasteLongTextToFileLen', label: 'Paste length to file', type: 'input' },
				{ key: 'pdfAsImage', label: 'Parse PDF as image instead of text', type: 'checkbox' }
			]
		},
		{
			title: 'Samplers',
			icon: Filter,
			fields: [
				{ key: 'samplers', label: 'Samplers queue', type: 'input' },
				{ key: 'dynatemp_range', label: 'Dynamic Temperature Range', type: 'input' },
				{ key: 'dynatemp_exponent', label: 'Dynamic Temperature Exponent', type: 'input' },
				{ key: 'typical_p', label: 'Typical P', type: 'input' },
				{ key: 'xtc_probability', label: 'XTC Probability', type: 'input' },
				{ key: 'xtc_threshold', label: 'XTC Threshold', type: 'input' }
			]
		},
		{
			title: 'Penalties',
			icon: Hand,
			fields: [
				{ key: 'repeat_last_n', label: 'Repeat Last N', type: 'input' },
				{ key: 'repeat_penalty', label: 'Repeat Penalty', type: 'input' },
				{ key: 'presence_penalty', label: 'Presence Penalty', type: 'input' },
				{ key: 'frequency_penalty', label: 'Frequency Penalty', type: 'input' },
				{ key: 'dry_multiplier', label: 'DRY Multiplier', type: 'input' },
				{ key: 'dry_base', label: 'DRY Base', type: 'input' },
				{ key: 'dry_allowed_length', label: 'DRY Allowed Length', type: 'input' },
				{ key: 'dry_penalty_last_n', label: 'DRY Penalty Last N', type: 'input' }
			]
		},
		{
			title: 'Reasoning',
			icon: MessageSquare,
			fields: [
				{
					key: 'showThoughtInProgress',
					label: 'Expand thought process by default when generating messages',
					type: 'checkbox'
				},
				{
					key: 'excludeThoughtOnReq',
					label: 'Exclude thought process when sending requests to API (Recommended for DeepSeek-R1)',
					type: 'checkbox'
				}
			]
		},
		{
			title: 'Advanced',
			icon: Plus,
			fields: [
				{ key: 'showTokensPerSecond', label: 'Show tokens per second', type: 'checkbox' },
				{ key: 'custom', label: 'Custom parameters (JSON format)', type: 'textarea' }
			]
		},
		{
			title: 'Experimental',
			icon: Beaker,
			fields: [
				{
					key: 'pyInterpreterEnabled',
					label: 'Enable Python interpreter',
					type: 'checkbox',
					help: 'This feature uses Pyodide to run Python code inside a Markdown code block. You will see a "Run" button on the code block, near the "Copy" button.'
				}
			]
		}
	];

	let currentSection = $derived(
		settingSections.find((section) => section.title === activeSection) || settingSections[0]
	);
</script>

<Dialog.Root {open} {onOpenChange}>
	<Dialog.Content class="flex h-[64vh] flex-col gap-0 p-0" style="max-width: 48rem;">
		<div class="flex flex-1 overflow-hidden">
			<div class="border-border/30 w-64 border-r p-6">
				<nav class="space-y-1 py-2">
					<Dialog.Title class="mb-6 flex items-center gap-2">Settings</Dialog.Title>

					{#each settingSections as section}
						<button
							class="hover:bg-accent flex w-full cursor-pointer items-center gap-3 rounded-lg px-3 py-2 text-left text-sm transition-colors {activeSection ===
							section.title
								? 'bg-accent text-accent-foreground'
								: 'text-muted-foreground'}"
							onclick={() => (activeSection = section.title)}
						>
							<section.icon class="h-4 w-4" />
							<span class="ml-2">{section.title}</span>
						</button>
					{/each}
				</nav>
			</div>

			<ScrollArea class="flex-1">
				<div class="space-y-6 p-6">
					<ChatSettingsSection title={currentSection.title} icon={currentSection.icon}>
						{#each currentSection.fields as field}
							<div class="space-y-2">
								{#if field.type === 'input'}
									<label for={field.key} class="block text-sm font-medium">
										{field.label}
									</label>

									<Input
										id={field.key}
										value={String(localConfig[field.key] || '')}
										onchange={(e) =>
											(localConfig[field.key] = e.currentTarget.value)}
										placeholder={`Default: ${defaultConfig[field.key] || 'none'}`}
										class="max-w-md"
									/>
									{#if field.help}
										<p class="text-muted-foreground mt-1 text-xs">
											{field.help}
										</p>
									{/if}
								{:else if field.type === 'textarea'}
									<label for={field.key} class="block text-sm font-medium">
										{field.label}
									</label>

									<Textarea
										id={field.key}
										value={String(localConfig[field.key] || '')}
										onchange={(e) =>
											(localConfig[field.key] = e.currentTarget.value)}
										placeholder={`Default: ${defaultConfig[field.key] || 'none'}`}
										class="min-h-[100px] max-w-2xl"
									/>
									{#if field.help}
										<p class="text-muted-foreground mt-1 text-xs">
											{field.help}
										</p>
									{/if}
								{:else if field.type === 'checkbox'}
									<div class="flex items-start space-x-3">
										<Checkbox
											id={field.key}
											checked={Boolean(localConfig[field.key])}
											onCheckedChange={(checked) =>
												(localConfig[field.key] = checked)}
											class="mt-1"
										/>

										<div class="space-y-1">
											<label
												for={field.key}
												class="cursor-pointer text-sm font-medium leading-none"
											>
												{field.label}
											</label>

											{#if field.help}
												<p class="text-muted-foreground text-xs">
													{field.help}
												</p>
											{/if}
										</div>
									</div>
								{/if}
							</div>
						{/each}
					</ChatSettingsSection>

					<div class="mt-8 border-t pt-6">
						<p class="text-muted-foreground text-xs">
							Settings are saved in browser's localStorage
						</p>
					</div>
				</div>
			</ScrollArea>
		</div>

		<ChatSettingsFooter onSave={handleSave} onReset={handleReset} onClose={handleClose} />
	</Dialog.Content>
</Dialog.Root>
