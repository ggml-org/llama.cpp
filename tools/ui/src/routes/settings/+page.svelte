<script lang="ts">
	import { SettingsChat } from '$lib/components/app/settings';
	import { APP_NAME } from '$lib/constants';
	import { toast } from 'svelte-sonner';
</script>

<svelte:head>
	<title>Settings - {APP_NAME}</title>
</svelte:head>

<!-- a phone's settings: the page the sidebar opens there, where a desktop or tablet opens the
     settings dialog. The section tabs pin right under the menu key, over the status bar fade;
     pinned, they fill the screen above them in the page's colour, so nothing scrolls through
     between the two. They stay under the menu key (z-25, where it is z-30 and the fade z-20) -->
<div
	class="[--settings-tabs-top:calc(var(--corner-control-top)+3rem)] flex w-full flex-col pt-(--page-top) max-md:pb-safe-4 md:pt-6"
>
	<div class="flex min-w-0 flex-col gap-1 px-4 pb-2 md:px-8">
		<h1 class="text-2xl font-semibold tracking-tight">Settings</h1>

		<p class="text-sm text-muted-foreground">How the app looks, chats and talks to the server.</p>
	</div>

	<SettingsChat
		mobileHeaderClass="top-(--settings-tabs-top) z-[25] data-stuck:before:absolute data-stuck:before:inset-x-0 data-stuck:before:bottom-full data-stuck:before:h-(--settings-tabs-top) data-stuck:before:bg-background"
		onClose={() => toast.success('Settings saved')}
	/>
</div>
