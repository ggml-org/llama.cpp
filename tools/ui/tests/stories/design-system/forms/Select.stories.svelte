<script lang="ts" module>
	import StateGallery from '../../helpers/StateGallery.svelte';
	import StateGalleryItem from '../../helpers/StateGalleryItem.svelte';
	import { defineMeta } from '@storybook/addon-svelte-csf';
	import * as Select from '$lib/components/ui/select';

	const { Story } = defineMeta({
		component: Select.Root,
		parameters: {
			layout: 'centered'
		},
		title: 'Design system/Forms/Select'
	});

	const themes = [
		{ label: 'System', value: 'system' },
		{ label: 'Light', value: 'light' },
		{ label: 'Dark', value: 'dark' }
	];
</script>

<Story asChild name="Default">
	<StateGallery columns={3} minItemWidth="12rem">
		<StateGalleryItem label="Placeholder">
			<Select.Root type="single">
				<Select.Trigger class="w-44">Choose a theme</Select.Trigger>
			</Select.Root>
		</StateGalleryItem>

		<StateGalleryItem label="Selected">
			<Select.Root type="single" value="dark">
				<Select.Trigger class="w-44">Dark</Select.Trigger>
			</Select.Root>
		</StateGalleryItem>

		<StateGalleryItem label="Disabled">
			<Select.Root disabled type="single" value="system">
				<Select.Trigger class="w-44">System</Select.Trigger>
			</Select.Root>
		</StateGalleryItem>

		<StateGalleryItem label="With options" note="click to open the list" span={3}>
			<div class="h-48">
				<Select.Root type="single" value="light">
					<Select.Trigger class="w-44">Light</Select.Trigger>

					<Select.Content>
						<Select.Group>
							<Select.Label>Theme</Select.Label>

							{#each themes as theme (theme.value)}
								<Select.Item label={theme.label} value={theme.value} />
							{/each}
						</Select.Group>
					</Select.Content>
				</Select.Root>
			</div>
		</StateGalleryItem>
	</StateGallery>
</Story>
