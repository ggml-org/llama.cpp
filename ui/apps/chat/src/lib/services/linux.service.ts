/**
 * LinuxService - Runs bash commands on a Linux machine inside the browser
 *
 * The machine is unkernel: a Linux kernel compiled to WebAssembly that runs
 * aarch64 programs on an emulated CPU. It loads the kernel of its release
 * and boots the Debian image at the configured URL on the first call, and
 * lives as long as the page. Its
 * root file system is an overlay on a persistent disk kept in IndexedDB:
 * the tab that holds the disk's Web Lock writes it, any other tab runs on a
 * copy in memory. Its TCP goes through llama-server's Wisp relay, when the
 * server runs with --ui-linux-net.
 * No reactive state; consumed by the agentic loop for the run_bash tool, and
 * by the settings to reset the machine.
 */

import {
	LINUX_DATA_DISK_LOCK,
	LINUX_DATA_DISK_NAME,
	LINUX_KERNEL_URL,
	LINUX_RESET_LOCKED_ERROR,
	LINUX_SHELL,
	LINUX_SHELL_COMMAND_FLAG,
	LINUX_SHELL_SCRIPT,
	LINUX_TIMEOUT_PROGRAM,
	LINUX_TIMEOUT_SECONDS_SUFFIX,
	LINUX_WISP_API_KEY_PARAM,
	LINUX_WISP_PATH,
	NEWLINE,
	RUN_BASH_EXIT_CODE_LINE,
	RUN_BASH_OUTPUT_MAX_CHARS,
	RUN_BASH_SIGNAL_EXIT_BASE,
	RUN_BASH_TIMED_OUT_MARK,
	RUN_BASH_TIMEOUT_EXIT_CODE,
	RUN_BASH_TIMEOUT_S_DEFAULT,
	RUN_BASH_TIMEOUT_S_MAX,
	RUN_BASH_TRUNCATION_NOTICE,
	UNKERNEL_DISK_KIND_DATA,
	UNKERNEL_DISK_MODE_PERSISTENT,
	UNKERNEL_DISK_MODE_SNAPSHOT,
	UNKERNEL_DISK_TEMPLATE_BUNDLED,
	UNKERNEL_NET_BACKEND_WISP,
	UNKERNEL_NET_KIND
} from '#lib/constants/index.ts';
import { UrlProtocol } from '#lib/enums/index.ts';
import { serverStore } from '#lib/stores/llama-server.svelte.ts';
import { settingsStore } from '#lib/stores/settings/index.svelte.ts';
import type { ToolExecutionResult } from '#lib/types/index.ts';
import { serverUrl } from '#lib/utils/server-config.ts';
import type { KernelInstance } from 'unkernel';

/** The machine, booting or running; null until the first call, and again after a failed boot. */
let machine: Promise<KernelInstance> | null = null;
/** Whether this tab holds the disk's Web Lock, asked once and then kept for the page's lifetime. */
let diskLock: Promise<boolean> | null = null;

function holdDiskLock(): Promise<boolean> {
	diskLock ??= new Promise((resolve) => {
		navigator.locks.request(LINUX_DATA_DISK_LOCK, { ifAvailable: true }, (lock) => {
			resolve(lock !== null);

			return lock ? new Promise<never>(() => {}) : undefined;
		});
	});

	return diskLock;
}

/** The server's Wisp relay as a WebSocket URL, with the API key a browser socket cannot send as a header. */
function wispUrl(): string | undefined {
	if (!serverStore.props?.ui_linux_net) return undefined;

	const url = new URL(serverUrl(LINUX_WISP_PATH), location.href);
	const apiKey = settingsStore.config.apiKey?.toString().trim();

	url.protocol =
		url.protocol === UrlProtocol.HTTPS ? UrlProtocol.WEBSOCKET_SECURE : UrlProtocol.WEBSOCKET;

	if (apiKey) url.searchParams.set(LINUX_WISP_API_KEY_PARAM, apiKey);

	return url.href;
}

async function boot(): Promise<KernelInstance> {
	const { checkSupport, createKernel } = await import('unkernel');
	const support = await checkSupport();

	if (!support.ok)
		throw new Error(`this browser cannot run the machine: ${support.problems.join(', ')}`);

	const imageUrl = String(settingsStore.config.linuxImageUrl);
	const wisp = wispUrl();
	const persistent = await holdDiskLock();

	return createKernel({
		disks: [
			{
				kind: UNKERNEL_DISK_KIND_DATA,
				mode: persistent ? UNKERNEL_DISK_MODE_PERSISTENT : UNKERNEL_DISK_MODE_SNAPSHOT,
				name: LINUX_DATA_DISK_NAME,
				template: UNKERNEL_DISK_TEMPLATE_BUNDLED
			}
		],
		distro: imageUrl,
		kernel: LINUX_KERNEL_URL,
		net: wisp
			? { backend: { kind: UNKERNEL_NET_BACKEND_WISP, servers: wisp }, kind: UNKERNEL_NET_KIND }
			: undefined
	});
}

function getMachine(): Promise<KernelInstance> {
	if (!machine) {
		machine = boot();
		machine.catch(() => {
			machine = null;
		});
	}

	return machine;
}

/** Keep the end of the output, where installs and builds report. */
function tail(output: string): string {
	if (output.length <= RUN_BASH_OUTPUT_MAX_CHARS) return output;

	return `${RUN_BASH_TRUNCATION_NOTICE}${NEWLINE}${output.slice(-RUN_BASH_OUTPUT_MAX_CHARS)}`;
}

export class LinuxService {
	/**
	 * Run a run_bash call: the command under coreutils' timeout, in the
	 * home directory, with stderr merged into stdout. The output
	 * ends with llama-server's exec_shell_command exit line.
	 */
	static async executeTool(
		params: Record<string, unknown>,
		signal?: AbortSignal
	): Promise<ToolExecutionResult> {
		const command = typeof params.command === 'string' ? params.command : '';

		if (!command) return { content: 'Missing required parameter: command', isError: true };

		const requested = Number(params.timeout);
		const timeout =
			Number.isFinite(requested) && requested > 0
				? Math.min(Math.ceil(requested), RUN_BASH_TIMEOUT_S_MAX)
				: RUN_BASH_TIMEOUT_S_DEFAULT;

		try {
			const vm = await getMachine();
			const result = await vm.exec(
				LINUX_TIMEOUT_PROGRAM,
				[
					`${timeout}${LINUX_TIMEOUT_SECONDS_SUFFIX}`,
					LINUX_SHELL,
					LINUX_SHELL_COMMAND_FLAG,
					LINUX_SHELL_SCRIPT,
					LINUX_SHELL,
					command
				],
				{ signal }
			);
			const code = result.exitCode ?? RUN_BASH_SIGNAL_EXIT_BASE + (result.signal ?? 0);

			let exitLine = RUN_BASH_EXIT_CODE_LINE(code);

			if (code === RUN_BASH_TIMEOUT_EXIT_CODE) exitLine += ` ${RUN_BASH_TIMED_OUT_MARK}`;

			return { content: `${tail(result.stdout)}${NEWLINE}${exitLine}`, isError: false };
		} catch (error) {
			return {
				content: `Error: ${error instanceof Error ? error.message : String(error)}`,
				isError: true
			};
		}
	}

	/**
	 * Stop the machine and empty its persistent disk: the next call boots a fresh image.
	 * Throws when another tab runs the machine on that disk.
	 */
	static async reset(): Promise<void> {
		if (!(await holdDiskLock())) throw new Error(LINUX_RESET_LOCKED_ERROR);

		const running = machine;

		machine = null;

		const vm = running ? await running.catch(() => null) : null;

		if (vm) {
			vm.stop();
			await vm.exited;
		}

		const { resetDataDisk } = await import('unkernel');

		await resetDataDisk({ name: LINUX_DATA_DISK_NAME });
	}
}
