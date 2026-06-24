let reopenCallback: (() => void) | null = null;

export function registerMcpDialogReopen(callback: () => void): void {
	reopenCallback = callback;
}

export function triggerMcpDialogReopen(): void {
	reopenCallback?.();
}
