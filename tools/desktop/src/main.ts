import { getForwardedServerArgs, getUserSpecifiedPort, hasModelsDir } from './args';
import { ServerProcess } from './server-process';
import { app, BrowserWindow, dialog } from 'electron';
import started from 'electron-squirrel-startup';
import getPort from 'get-port';

// handle creating/removing shortcuts on Windows when installing/uninstalling
if (started) {
	app.quit();
}

async function promptForModelsDir(): Promise<string | null> {
	const title = 'Select your models directory';
	const result = await dialog.showOpenDialog({
		message: title,
		properties: ['openDirectory'],
		title
	});

	return result.canceled ? null : result.filePaths[0];
}

async function startServer(): Promise<ServerProcess | null> {
	const serverArgs = getForwardedServerArgs();

	if (!hasModelsDir()) {
		const modelsDir = await promptForModelsDir();

		if (!modelsDir) return null;

		serverArgs.push('--models-dir', modelsDir);
	}

	let port = await getUserSpecifiedPort();

	if (!port) {
		port = await getPort();
		serverArgs.push('--port', port.toString());
	}

	const serverProcess = new ServerProcess(port, serverArgs);

	await serverProcess.waitUntilRunning();

	return serverProcess;
}

async function createWindow(serverProcess: ServerProcess) {
	const window = new BrowserWindow({
		height: 600,
		show: false,
		title: 'llama.cpp',
		width: 800
	});

	window.loadURL(serverProcess.getUrl());

	window.on('ready-to-show', () => {
		window.show();
	});

	window.on('closed', () => {
		serverProcess.stop();
	});
}

let launching = false;

async function launchApp() {
	if (launching) return;

	launching = true;

	try {
		const serverProcess = await startServer();

		if (!serverProcess) return;

		createWindow(serverProcess);
	} catch (error) {
		console.error(error);

		dialog.showErrorBox(
			'Error',
			`Failed to start app: ${error instanceof Error ? error.message : String(error)}`
		);
	} finally {
		launching = false;
	}
}

app.setAppUserModelId('com.squirrel.llama.llama');

app.whenReady().then(launchApp);

app.on('window-all-closed', () => {
	// on macOS, it's common to keep the app running when the window is closed
	if (process.platform !== 'darwin') {
		app.quit();
	}
});

app.on('activate', () => {
	if (BrowserWindow.getAllWindows().length === 0) {
		launchApp();
	}
});
