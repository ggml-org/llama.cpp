import { BuiltInTool, JsonSchemaType, ToolCallType } from '#lib/enums/index.ts';
import type { OpenAIToolDefinition } from '#lib/types/index.ts';

export const RUN_BASH_TOOL_NAME = BuiltInTool.BROWSER_RUN_BASH;

/** Seconds a command may run. The CPU is emulated, so the default leaves room for apt and builds. */
export const RUN_BASH_TIMEOUT_S_DEFAULT = 120;

export const RUN_BASH_TIMEOUT_S_MAX = 1800;

/** Characters of output the model gets: the end is kept, where installs and builds report. */
export const RUN_BASH_OUTPUT_MAX_CHARS = 16384;

export const RUN_BASH_TRUNCATION_NOTICE = '[output truncated, its end follows]';

/** Exit status of coreutils' timeout when the command ran out of time. */
export const RUN_BASH_TIMEOUT_EXIT_CODE = 124;

/** Exit status a shell reports for a command killed by a signal: this base plus the signal number. */
export const RUN_BASH_SIGNAL_EXIT_BASE = 128;

/** The exit line llama-server's exec_shell_command ends its output with, and its timeout mark. */
export const RUN_BASH_EXIT_CODE_LINE = (code: number) => `[exit code: ${code}]`;

export const RUN_BASH_TIMED_OUT_MARK = '[exit due to timed out]';

/**
 * The command line: coreutils' timeout around bash, whose script merges
 * stderr into stdout and runs the model's command, passed as $1.
 */
export const LINUX_TIMEOUT_PROGRAM = 'timeout';

export const LINUX_TIMEOUT_SECONDS_SUFFIX = 's';

export const LINUX_SHELL = 'bash';

export const LINUX_SHELL_COMMAND_FLAG = '-c';

export const LINUX_SHELL_SCRIPT = 'exec 2>&1; eval "$1"';

/**
 * The unkernel release the browser Linux machine runs, mirrored unchanged in a Hugging Face bucket
 * (CORS and Range requests): its kernel and its Debian image. It matches the unkernel SDK's version.
 */
export const LINUX_RELEASE_URL =
	'https://huggingface.co/buckets/ggml-org-dev/unkernel/resolve/v0.2.0-261119-97d7945';

export const LINUX_KERNEL_URL = `${LINUX_RELEASE_URL}/unkernel.wasm`;

export const LINUX_IMAGE_URL_DEFAULT = `${LINUX_RELEASE_URL}/debian.erofs`;

/** The persistent disk's IndexedDB store, database "kwdisk:<name>-<version>". */
export const LINUX_DATA_DISK_NAME = 'llama-ui-linux';

/** llama-server's Wisp relay (`--ui-linux-net`) and the query parameter that carries the API key to it. */
export const LINUX_WISP_PATH = '/wisp/';

export const LINUX_WISP_API_KEY_PARAM = 'api_key';

/** Why a reset waits: the tab that runs the machine on the persistent disk is another one. */
export const LINUX_RESET_LOCKED_ERROR =
	'another tab runs the Linux machine on its disk: close it, then reset again';

/** Web Lock of the persistent disk: the tab that holds it writes the disk, others run on a copy. */
export const LINUX_DATA_DISK_LOCK = 'llama-ui-linux-disk';

/** Spec values of the unkernel SDK (DiskSpec, NetSpec). */
export const UNKERNEL_DISK_KIND_DATA = 'data';

export const UNKERNEL_DISK_TEMPLATE_BUNDLED = 'bundled';

export const UNKERNEL_DISK_MODE_PERSISTENT = 'persistent';

export const UNKERNEL_DISK_MODE_SNAPSHOT = 'snapshot';

export const UNKERNEL_NET_KIND = 'net';

export const UNKERNEL_NET_BACKEND_WISP = 'wisp';

export function buildRunBashToolDefinition(): OpenAIToolDefinition {
	return {
		function: {
			description:
				'Run a bash command on a Debian Linux machine that runs inside this browser (an emulated aarch64 CPU: fine for shell work and scripts, slow for heavy compute). Commands start in the home directory, and the disk persists across chats, so packages installed with apt-get stay. The network, when there is one, carries TCP only (no UDP, no ping). Returns stdout and stderr combined, then the exit code.',
			name: RUN_BASH_TOOL_NAME,
			parameters: {
				properties: {
					command: {
						description: 'Bash command to run',
						type: JsonSchemaType.STRING
					},
					description: {
						description:
							"Short plain-language description of what this command does, starting with a verb, e.g. 'Install the build tools'. 3 to 10 words. Shown to the user.",
						type: JsonSchemaType.STRING
					},
					timeout: {
						description: `Timeout in seconds (default ${RUN_BASH_TIMEOUT_S_DEFAULT}, max ${RUN_BASH_TIMEOUT_S_MAX})`,
						type: JsonSchemaType.NUMBER
					}
				},
				required: ['description', 'command'],
				type: JsonSchemaType.OBJECT
			}
		},
		type: ToolCallType.FUNCTION
	};
}
