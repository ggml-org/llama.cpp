const OPTION_BUILD_DIR = '--build-dir';
const args = process.argv.slice(2);

export function hasModelsDir(): boolean {
	return args.includes('--models-dir') || !!process.env.LLAMA_ARG_MODELS_DIR;
}

export function parsePort(port: string): number | null {
	const portNumber = parseInt(port, 10);

	return Number.isSafeInteger(portNumber) && portNumber >= 1 && portNumber <= 65535
		? portNumber
		: null;
}

export async function getUserSpecifiedPort(): Promise<number | null> {
	const portIndex = args.indexOf('--port');

	if (portIndex >= 0) {
		if (portIndex + 1 < args.length) {
			const port = parsePort(args[portIndex + 1]);

			if (port) return port;
		}

		throw new Error('Valid port number is required after --port');
	}

	const envPort = process.env.LLAMA_ARG_PORT;

	if (envPort) {
		const port = parsePort(envPort);

		if (port) return port;

		throw new Error('LLAMA_ARG_PORT must be a valid port number');
	}

	return null;
}

export function getForwardedServerArgs(): string[] {
	const serverArgs = [...args];
	// Remove --build-dir development option
	const buildDirIndex = serverArgs.indexOf(OPTION_BUILD_DIR);

	if (buildDirIndex >= 0) {
		serverArgs.splice(buildDirIndex, 2);
	}

	return serverArgs;
}

/**
 * Used in development only
 */
export function getLlamaBuildDir(): string {
	const buildDirIndex = args.indexOf(OPTION_BUILD_DIR);

	if (buildDirIndex < 0 || buildDirIndex + 1 >= args.length)
		throw new Error(`Launch with \`npm start -- -- ${OPTION_BUILD_DIR} <cmake-build-dir>\``);

	return args[buildDirIndex + 1];
}
