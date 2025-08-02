<script lang="ts">
	import * as Dialog from '$lib/components/ui/dialog';
	import { Settings, Filter, Hand, MessageSquare, Plus, Beaker } from '@lucide/svelte';
	import ChatSettingsSidebar from './ChatSettingsSidebar.svelte';
	import ChatSettingsContent from './ChatSettingsContent.svelte';
	import ChatSettingsFooter from './ChatSettingsFooter.svelte';
	import type { SettingSection, SettingsConfig } from '$lib/types/settings';

	interface Props {
		open?: boolean;
		onOpenChange?: (open: boolean) => void;
	}

	let { open = false, onOpenChange }: Props = $props();

	// Mock configuration - will be replaced with real stores
	let config: SettingsConfig = $state({
		// General
		apiKey: '',
		systemMessage: '',
		temperature: 0.8,
		top_k: 40,
		top_p: 0.95,
		min_p: 0.05,
		max_tokens: 2048,
		pasteLongTextToFileLen: 2000,
		pdfAsImage: false,

		// Samplers
		samplers: 'top_k;tfs_z;typical_p;top_p;min_p;temperature',
		dynatemp_range: 0.0,
		dynatemp_exponent: 1.0,
		typical_p: 1.0,
		xtc_probability: 0.0,
		xtc_threshold: 0.1,

		// Penalties
		repeat_last_n: 64,
		repeat_penalty: 1.0,
		presence_penalty: 0.0,
		frequency_penalty: 0.0,
		dry_multiplier: 0.0,
		dry_base: 1.75,
		dry_allowed_length: 2,
		dry_penalty_last_n: -1,

		// Reasoning
		showThoughtInProgress: true,
		excludeThoughtOnReq: false,

		// Advanced
		showTokensPerSecond: false,
		custom: '',

		// Experimental
		pyInterpreterEnabled: false
	});

	const defaultConfig: SettingsConfig = $state.snapshot(config);

	let activeSection = $state('General');

	const settingSections: SettingSection[] = [
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
					label: 'Show thought in progress',
					type: 'checkbox'
				},
				{
					key: 'excludeThoughtOnReq',
					label: 'Exclude thought on request',
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

	// Get current section
	const currentSection = $derived(
		settingSections.find((section) => section.title === activeSection) || settingSections[0]
	);

	function handleSave() {
		// TODO: Save to stores and localStorage
		console.log('Saving config:', config);
		onOpenChange?.(false);
	}

	function handleReset() {
		Object.assign(config, defaultConfig);
	}

	function handleClose() {
		onOpenChange?.(false);
	}

	function handleSectionChange(section: string) {
		activeSection = section;
	}

	function handleConfigChange(key: string, value: any) {
		config[key as keyof SettingsConfig] = value;
	}
</script>

<Dialog.Root {open} {onOpenChange}>
	<Dialog.Content class="flex h-[64vh] flex-col" style="max-width: 48rem;">
		<Dialog.Header>
			<Dialog.Title class="flex items-center gap-2">
				<Settings class="h-5 w-5" />
				Settings
			</Dialog.Title>
		</Dialog.Header>

		<!-- Content with Sidebar Layout -->
		<div class="flex flex-1 overflow-hidden">
			<ChatSettingsSidebar
				sections={settingSections}
				{activeSection}
				onSectionChange={handleSectionChange}
			/>

			<ChatSettingsContent
				{currentSection}
				{config}
				{defaultConfig}
				onConfigChange={handleConfigChange}
			/>
		</div>

		<ChatSettingsFooter onSave={handleSave} onReset={handleReset} onClose={handleClose} />
	</Dialog.Content>
</Dialog.Root>
