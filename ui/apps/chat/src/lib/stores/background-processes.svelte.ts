/**
 * backgroundProcessesStore - Long-lived commands the agent left running
 *
 * Lists the processes `exec_shell_command` started with `background: true`
 * (dev servers, watchers) from the server's `/tools/background`, and stops
 * them. Consumed by the chat form's background processes chip and dialog.
 */

import { BACKGROUND_PROCESSES_POLL_MS } from '#lib/constants/index.ts';
import { BuiltInTool } from '#lib/enums/index.ts';
import { ToolsService } from '#lib/services/tools.service.ts';
import { agenticMode } from '#lib/stores/agentic-mode.svelte.ts';
import { toolsStore } from '#lib/stores/tools.svelte.ts';
import type { BackgroundProcess } from '#lib/types/index.ts';

class BackgroundProcessesStore {
	private _processes = $state<BackgroundProcess[]>([]);

	dialogOpen = $state(false);

	get processes(): BackgroundProcess[] {
		return this._processes;
	}

	get runningCount(): number {
		return this._processes.filter((p) => p.running).length;
	}

	/** The server runs shell commands, so the agent may have started some */
	get available(): boolean {
		return agenticMode.enabled && toolsStore.hasServerTool(BuiltInTool.SERVER_EXEC_SHELL_COMMAND);
	}

	async refresh(): Promise<void> {
		if (!this.available) {
			this._processes = [];

			return;
		}

		try {
			this._processes = await ToolsService.listBackgroundProcesses();
		} catch {
			// an older server without the endpoint, or one that went away: show nothing
			this._processes = [];
		}
	}

	async stop(id: number): Promise<void> {
		// leave the list at once, the server takes a moment to stop the process group
		this._processes = this._processes.filter((p) => p.id !== id);

		try {
			await ToolsService.stopBackgroundProcess(id);
		} finally {
			await this.refresh();
		}
	}

	async stopAll(): Promise<void> {
		await Promise.all(this._processes.map((p) => this.stop(p.id)));
	}

	/** Polls while mounted; returns the cleanup for an `$effect` */
	watch(): () => void {
		void this.refresh();
		const timer = setInterval(() => void this.refresh(), BACKGROUND_PROCESSES_POLL_MS);

		return () => clearInterval(timer);
	}
}

export const backgroundProcessesStore = new BackgroundProcessesStore();
