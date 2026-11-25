/**
 * Storage targets for demo scripts and videos, picked by the scheme of the
 * DEMO_STORAGE URI:
 *
 *   hf://buckets/<owner>/<bucket>[/prefix]   Hugging Face bucket, via the `hf` CLI login
 *   file:///some/dir                         local or mounted directory
 */

import { execFileSync } from 'node:child_process';
import { copyFileSync, cpSync, existsSync, mkdirSync } from 'node:fs';
import { dirname, join } from 'node:path';
import { fileURLToPath } from 'node:url';

export interface DemoStorage {
	/** Copies every file under remote `dir` into local `target`; returns false when it has none. */
	downloadDir(dir: string, target: string): boolean;
	/** Copies local `file` to remote `key`. */
	upload(file: string, key: string): void;
}

function joinKey(base: string, key: string): string {
	return `${base.replace(/\/+$/, '')}/${key}`;
}

// plain fields: the CLI runs under Node's type stripping, which has no parameter properties
class HfStorage implements DemoStorage {
	private readonly base: string;

	constructor(base: string) {
		this.base = base;
	}

	downloadDir(dir: string, target: string): boolean {
		mkdirSync(target, { recursive: true });

		try {
			execFileSync('hf', ['buckets', 'sync', joinKey(this.base, dir), target], {
				stdio: 'inherit'
			});
		} catch {
			return false;
		}

		return true;
	}

	upload(file: string, key: string): void {
		execFileSync('hf', ['buckets', 'cp', file, joinKey(this.base, key)], { stdio: 'inherit' });
	}
}

class FileStorage implements DemoStorage {
	private readonly root: string;

	constructor(root: string) {
		this.root = root;
	}

	downloadDir(dir: string, target: string): boolean {
		const source = join(this.root, dir);

		if (!existsSync(source)) return false;

		cpSync(source, target, { recursive: true });

		return true;
	}

	upload(file: string, key: string): void {
		const target = join(this.root, key);

		mkdirSync(dirname(target), { recursive: true });
		copyFileSync(file, target);
	}
}

export function getDemoStorage(uri = process.env.DEMO_STORAGE): DemoStorage | null {
	if (!uri) return null;

	if (uri.startsWith('hf://')) return new HfStorage(uri);

	if (uri.startsWith('file://')) return new FileStorage(fileURLToPath(uri));

	throw new Error(`DEMO_STORAGE: unsupported scheme in "${uri}" (use hf:// or file://)`);
}

/** Public URL of a stored key, when DEMO_STORAGE_PUBLIC_URL is set. */
export function getPublicUrl(key: string): string | null {
	const base = process.env.DEMO_STORAGE_PUBLIC_URL;

	return base ? joinKey(base, key) : null;
}
