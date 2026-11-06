// OS mockups for the stories: a desktop's wallpaper, and on it the macOS Dock and menu bar or the Windows taskbar,
// to picture the brand's icons where the apps put them
export {
	default as MacDock,
	PLACEHOLDERS as MAC_DOCK_PLACEHOLDERS,
	type DockApp
} from './MacDock.svelte';
export { default as MacMenuBar } from './MacMenuBar.svelte';
export { default as MacStatusIcon } from './MacStatusIcon.svelte';
export { default as MockDesktop } from './MockDesktop.svelte';
export {
	default as WindowsTaskbar,
	PLACEHOLDERS as WINDOWS_TASKBAR_PLACEHOLDERS,
	type TaskbarApp
} from './WindowsTaskbar.svelte';
