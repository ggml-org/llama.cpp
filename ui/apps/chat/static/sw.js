// Older builds registered the service worker at ./sw.js. Their update check lands here,
// so remove that worker and its caches; the reloaded page registers service-worker.js.
self.addEventListener('install', () => self.skipWaiting());
self.addEventListener('activate', (e) =>
	e.waitUntil(
		(async () => {
			await self.registration.unregister();
			for (const key of await caches.keys()) await caches.delete(key);
			for (const c of await self.clients.matchAll({ type: 'window' })) c.navigate(c.url);
		})()
	)
);
