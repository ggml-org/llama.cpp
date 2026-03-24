/// llama.cpp PWA Service Worker
/// Provides offline caching for the single-page application.

const CACHE_NAME = 'llama-cpp-v1';
const STATIC_ASSETS = ['./'];

self.addEventListener('install', (event) => {
  event.waitUntil(
    caches.open(CACHE_NAME).then((cache) => cache.addAll(STATIC_ASSETS))
  );
  self.skipWaiting();
});

self.addEventListener('activate', (event) => {
  event.waitUntil(
    caches.keys().then((keys) =>
      Promise.all(keys.filter((k) => k !== CACHE_NAME).map((k) => caches.delete(k)))
    )
  );
  self.clients.claim();
});

self.addEventListener('fetch', (event) => {
  const { request } = event;

  // Only cache GET requests
  if (request.method !== 'GET') return;

  // Skip API calls - always go to network
  const url = new URL(request.url);
  if (
    url.pathname.startsWith('/v1/') ||
    url.pathname === '/health' ||
    url.pathname === '/props' ||
    url.pathname === '/models' ||
    url.pathname === '/cors-proxy'
  ) {
    return;
  }

  // Network-first for navigation (the HTML), falling back to cache
  event.respondWith(
    fetch(request)
      .then((response) => {
        const clone = response.clone();
        caches.open(CACHE_NAME).then((cache) => cache.put(request, clone));
        return response;
      })
      .catch(() => caches.match(request))
  );
});
