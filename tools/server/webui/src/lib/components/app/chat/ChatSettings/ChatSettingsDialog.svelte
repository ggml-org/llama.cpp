<script lang="ts">
	import { Settings, Funnel, AlertTriangle, Brain, Cog, Monitor, Sun, Moon } from '@lucide/svelte';
	import { ChatSettingsFooter, ChatSettingsSection } from '$lib/components/app';
	import { Checkbox } from '$lib/components/ui/checkbox';
	import * as Dialog from '$lib/components/ui/dialog';
	import { Input } from '$lib/components/ui/input';
	import Label from '$lib/components/ui/label/label.svelte';
	import { ScrollArea } from '$lib/components/ui/scroll-area';
	import * as Select from '$lib/components/ui/select';
	import { Textarea } from '$lib/components/ui/textarea';
	import { SETTING_CONFIG_DEFAULT, SETTING_CONFIG_INFO } from '$lib/constants/settings-config';
	import { supportsVision } from '$lib/stores/server.svelte';
	import { config, updateMultipleConfig, resetConfig } from '$lib/stores/settings.svelte';
	import { setMode } from 'mode-watcher';
	import type { Component } from 'svelte';

	interface Props {
		onOpenChange?: (open: boolean) => void;
		open?: boolean;
	}

	let { onOpenChange, open = false }: Props = $props();

	const settingSections: Array<{
		fields: SettingsFieldConfig[];
		icon: Component;
		title: string;
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
				{
					key: 'theme',
					label: 'Theme',
					type: 'select',
					options: [
						{ value: 'system', label: 'System', icon: Monitor },
						{ value: 'light', label: 'Light', icon: Sun },
						{ value: 'dark', label: 'Dark', icon: Moon }
					]
				},
				{
					key: 'showTokensPerSecond',
					label: 'Show tokens per second',
					type: 'checkbox'
				},
				{
					key: 'keepStatsVisible',
					label: 'Keep stats visible after generation',
					type: 'checkbox'
				},
				{
					key: 'askForTitleConfirmation',
					label: 'Ask for confirmation before changing conversation title',
					type: 'checkbox'
				},
				{
					key: 'pasteLongTextToFileLen',
					label: 'Paste long text to file length',
					type: 'input'
				},
				{
					key: 'pdfAsImage',
					label: 'Parse PDF as image',
					type: 'checkbox'
				}
			]
		},
		{
			title: 'Samplers',
			icon: Funnel,
			fields: [
				{
					key: 'samplers',
					label: 'Samplers',
					type: 'input'
				}
			]
		},
		{
			title: 'Penalties',
			icon: AlertTriangle,
			fields: [
				{
					key: 'repeat_last_n',
					label: 'Repeat last N',
					type: 'input'
				},
				{
					key: 'repeat_penalty',
					label: 'Repeat penalty',
					type: 'input'
				},
				{
					key: 'presence_penalty',
					label: 'Presence penalty',
					type: 'input'
				},
				{
					key: 'frequency_penalty',
					label: 'Frequency penalty',
					type: 'input'
				},
				{
					key: 'dry_multiplier',
					label: 'DRY multiplier',
					type: 'input'
				},
				{
					key: 'dry_base',
					label: 'DRY base',
					type: 'input'
				},
				{
					key: 'dry_allowed_length',
					label: 'DRY allowed length',
					type: 'input'
				},
				{
					key: 'dry_penalty_last_n',
					label: 'DRY penalty last N',
					type: 'input'
				}
			]
		},
		{
			title: 'Reasoning',
			icon: Brain,
			fields: [
				{
					key: 'showThoughtInProgress',
					label: 'Show thought in progress',
					type: 'checkbox'
				}
			]
		},
		{
			title: 'Advanced',
			icon: Cog,
			fields: [
				{
					key: 'temperature',
					label: 'Temperature',
					type: 'input'
				},
				{
					key: 'dynatemp_range',
					label: 'Dynamic temperature range',
					type: 'input'
				},
				{
					key: 'dynatemp_exponent',
					label: 'Dynamic temperature exponent',
					type: 'input'
				},
				{
					key: 'top_k',
					label: 'Top K',
					type: 'input'
				},
				{
					key: 'top_p',
					label: 'Top P',
					type: 'input'
				},
				{
					key: 'min_p',
					label: 'Min P',
					type: 'input'
				},
				{
					key: 'xtc_probability',
					label: 'XTC probability',
					type: 'input'
				},
				{
					key: 'xtc_threshold',
					label: 'XTC threshold',
					type: 'input'
				},
				{
					key: 'typ_p',
					label: 'Typical P',
					type: 'input'
				},
				{
					key: 'max_tokens',
					label: 'Max tokens',
					type: 'input'
				},
				{
					key: 'custom',
					label: 'Custom JSON',
					type: 'textarea'
				}
			]
		}
		// TODO: Experimental features section will be implemented after initial release
		// This includes Python interpreter (Pyodide integration) and other experimental features
		// {
		// 	title: 'Experimental',
		// 	icon: Beaker,
		// 	fields: [
		// 		{
		// 			key: 'pyInterpreterEnabled',
		// 			label: 'Enable Python interpreter',
		// 			type: 'checkbox'
		// 		}
		// 	]
		// }
	];

	let activeSection = $state('General');
	let currentSection = $derived(
		settingSections.find((section) => section.title === activeSection) || settingSections[0]
	);
	let localConfig: SettingsConfigType = $state({ ...config() });
	let originalTheme: string = $state('');

	function handleThemeChange(newTheme: string) {
		localConfig.theme = newTheme;

		setMode(newTheme as 'light' | 'dark' | 'system');
	}

	function handleClose() {
		if (localConfig.theme !== originalTheme) {
			setMode(originalTheme as 'light' | 'dark' | 'system');
		}
		onOpenChange?.(false);
	}

	function handleReset() {
		resetConfig();

		localConfig = { ...SETTING_CONFIG_DEFAULT };

		setMode(SETTING_CONFIG_DEFAULT.theme as 'light' | 'dark' | 'system');
		originalTheme = SETTING_CONFIG_DEFAULT.theme as string;
	}

	function handleSave() {
		// Validate custom JSON if provided
		if (localConfig.custom && typeof localConfig.custom === 'string' && localConfig.custom.trim()) {
			try {
				JSON.parse(localConfig.custom);
			} catch (error) {
				alert('Invalid JSON in custom parameters. Please check the format and try again.');
				console.error(error);
				return;
			}
		}

		// Convert numeric strings to numbers for numeric fields
		const processedConfig = { ...localConfig };
		const numericFields = [
			'temperature',
			'top_k',
			'top_p',
			'min_p',
			'max_tokens',
			'pasteLongTextToFileLen',
			'dynatemp_range',
			'dynatemp_exponent',
			'typ_p',
			'xtc_probability',
			'xtc_threshold',
			'repeat_last_n',
			'repeat_penalty',
			'presence_penalty',
			'frequency_penalty',
			'dry_multiplier',
			'dry_base',
			'dry_allowed_length',
			'dry_penalty_last_n'
		];

		for (const field of numericFields) {
			if (processedConfig[field] !== undefined && processedConfig[field] !== '') {
				const numValue = Number(processedConfig[field]);
				if (!isNaN(numValue)) {
					processedConfig[field] = numValue;
				} else {
					alert(`Invalid numeric value for ${field}. Please enter a valid number.`);
					return;
				}
			}
		}

		updateMultipleConfig(processedConfig);
		onOpenChange?.(false);
	}

	$effect(() => {
		if (open) {
			localConfig = { ...config() };
			originalTheme = config().theme as string;
		}
	});
</script>

<Dialog.Root {open} onOpenChange={handleClose}>
	<Dialog.Content class="flex h-[64vh] flex-col gap-0 p-0" style="max-width: 48rem;">
		<div class="flex flex-1 overflow-hidden">
			<div class="w-64 border-r border-border/30 p-6">
				<nav class="space-y-1 py-2">
					<Dialog.Title class="mb-6 flex items-center gap-2">Settings</Dialog.Title>

					{#each settingSections as section (section.title)}
						<button
							class="flex w-full cursor-pointer items-center gap-3 rounded-lg px-3 py-2 text-left text-sm transition-colors hover:bg-accent {activeSection ===
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
					<ChatSettingsSection title={currentSection.title} Icon={currentSection.icon}>
						{#each currentSection.fields as field (field.key)}
							<div class="space-y-2">
								{#if field.type === 'input'}
									<Label for={field.key} class="block text-sm font-medium">
										{field.label}
									</Label>

									<Input
										id={field.key}
										value={String(localConfig[field.key] || '')}
										onchange={(e) => (localConfig[field.key] = e.currentTarget.value)}
										placeholder={`Default: ${SETTING_CONFIG_DEFAULT[field.key] || 'none'}`}
										class="max-w-md"
									/>
									{#if field.help || SETTING_CONFIG_INFO[field.key]}
										<p class="mt-1 text-xs text-muted-foreground">
											{field.help || SETTING_CONFIG_INFO[field.key]}
										</p>
									{/if}
								{:else if field.type === 'textarea'}
									<Label for={field.key} class="block text-sm font-medium">
										{field.label}
									</Label>

									<Textarea
										id={field.key}
										value={String(localConfig[field.key] || '')}
										onchange={(e) => (localConfig[field.key] = e.currentTarget.value)}
										placeholder={`Default: ${SETTING_CONFIG_DEFAULT[field.key] || 'none'}`}
										class="min-h-[100px] max-w-2xl"
									/>
									{#if field.help || SETTING_CONFIG_INFO[field.key]}
										<p class="mt-1 text-xs text-muted-foreground">
											{field.help || SETTING_CONFIG_INFO[field.key]}
										</p>
									{/if}
								{:else if field.type === 'select'}
									{@const selectedOption = field.options?.find(
										(opt: { value: string; label: string; icon?: Component }) =>
											opt.value === localConfig[field.key]
									)}

									<Label for={field.key} class="block text-sm font-medium">
										{field.label}
									</Label>

									<Select.Root
										type="single"
										value={localConfig[field.key]}
										onValueChange={(value) => {
											if (field.key === 'theme' && value) {
												handleThemeChange(value);
											} else {
												localConfig[field.key] = value;
											}
										}}
									>
										<Select.Trigger class="max-w-md">
											<div class="flex items-center gap-2">
												{#if selectedOption?.icon}
													{@const IconComponent = selectedOption.icon}
													<IconComponent class="h-4 w-4" />
												{/if}

												{selectedOption?.label || `Select ${field.label.toLowerCase()}`}
											</div>
										</Select.Trigger>
										<Select.Content>
											{#if field.options}
												{#each field.options as option (option.value)}
													<Select.Item value={option.value} label={option.label}>
														<div class="flex items-center gap-2">
															{#if option.icon}
																{@const IconComponent = option.icon}
																<IconComponent class="h-4 w-4" />
															{/if}
															{option.label}
														</div>
													</Select.Item>
												{/each}
											{/if}
										</Select.Content>
									</Select.Root>
									{#if field.help || SETTING_CONFIG_INFO[field.key]}
										<p class="mt-1 text-xs text-muted-foreground">
											{field.help || SETTING_CONFIG_INFO[field.key]}
										</p>
									{/if}
								{:else if field.type === 'checkbox'}
									{@const isDisabled = field.key === 'pdfAsImage' && !supportsVision()}
									<div class="flex items-start space-x-3">
										<Checkbox
											id={field.key}
											checked={Boolean(localConfig[field.key])}
											disabled={isDisabled}
											onCheckedChange={(checked) => (localConfig[field.key] = checked)}
											class="mt-1"
										/>

										<div class="space-y-1">
											<label
												for={field.key}
												class="cursor-pointer text-sm leading-none font-medium {isDisabled
													? 'text-muted-foreground'
													: ''}"
											>
												{field.label}
											</label>

											{#if field.help || SETTING_CONFIG_INFO[field.key]}
												<p class="text-xs text-muted-foreground">
													{field.help || SETTING_CONFIG_INFO[field.key]}
												</p>
											{:else if field.key === 'pdfAsImage' && !supportsVision()}
												<p class="text-xs text-muted-foreground">
													PDF-to-image processing requires a vision-capable model. PDFs will be
													processed as text.
												</p>
											{/if}
										</div>
									</div>
								{/if}
							</div>
						{/each}
					</ChatSettingsSection>

					<div class="mt-8 border-t pt-6">
						<p class="text-xs text-muted-foreground">
							Settings are saved in browser's localStorage
						</p>
					</div>
				</div>
			</ScrollArea>
		</div>

		<ChatSettingsFooter onClose={handleClose} onReset={handleReset} onSave={handleSave} />
	</Dialog.Content>
</Dialog.Root>
