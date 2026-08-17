import { getLlamaBuildDir } from './args';
import { app } from 'electron';
import type { ChildProcess } from 'node:child_process';
import { spawn } from 'node:child_process';
import { connect } from 'node:net';
import path from 'node:path';

export class ServerProcess {
	private static readonly HOST = '127.0.0.1';
	private readonly port: number;
	private process: ChildProcess;

	constructor(port: number, args: string[]) {
		this.port = port;

		this.process = spawn(ServerProcess.getExecutablePath(), args);

		this.process.stdout?.on('data', (data) => {
			console.log(data.toString());
		});
		this.process.stderr?.on('data', (data) => {
			console.error(data.toString());
		});
		this.process.on('error', (error) => {
			console.error(`llama-server process error: ${error}`);
		});
		this.process.on('close', (code) => {
			console.log(`llama-server process exited with code ${code}`);
		});
	}

	public stop() {
		if (this.isStopped()) return;

		if (!this.process.kill()) throw new Error('Failed to kill llama-server process');
	}

	public getUrl(): string {
		return `http://${ServerProcess.HOST}:${this.port}`;
	}

	public isStopped() {
		return this.process.exitCode !== null;
	}

	public async isServerUp() {
		return new Promise<boolean>((resolve) => {
			const socket = connect(this.port, ServerProcess.HOST);

			socket.on('connect', () => {
				socket.destroy();
				resolve(true);
			});

			socket.on('error', () => {
				socket.destroy();
				resolve(false);
			});
		});
	}

	public async waitUntilRunning() {
		while (true) {
			if (this.isStopped())
				throw new Error(`llama-server process exited with code ${this.process.exitCode}`);

			if (await this.isServerUp()) return;

			await new Promise((resolve) => setTimeout(resolve, 200));
		}
	}

	private static getExecutablePath(): string {
		const dir = app.isPackaged ? process.resourcesPath : path.join(getLlamaBuildDir(), 'bin');
		const name = process.platform === 'win32' ? 'llama-server.exe' : 'llama-server';

		return path.join(dir, name);
	}
}
