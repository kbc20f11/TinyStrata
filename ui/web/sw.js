// Retire the previous local chat UI's root service worker. TinyStrata uses live
// API responses; keep chats/settings and existing caches untouched.
self.addEventListener("install", () => self.skipWaiting());
self.addEventListener("activate", (event) => {
  event.waitUntil((async () => {
    await self.clients.claim();
    await self.registration.unregister();
  })());
});
