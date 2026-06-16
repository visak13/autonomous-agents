import { defineConfig } from "vite";
import react from "@vitejs/plugin-react";

// The backend serves GET / -> static/index.html and mounts /static -> static/.
// So the built SPA must (a) base all asset URLs under /static/ and (b) emit the
// production bundle straight INTO the backend's static dir, overwriting the
// previous index.html in place (the React SPA fully supersedes the round-1 vanilla UI).
//
// outDir is OUTSIDE this Vite root, so Vite refuses to clear it unless we opt in
// with emptyOutDir. We DO want it cleared: any stale app.js/styles.css are no
// longer referenced by the new index.html, so leaving them would be orphaned
// artifacts.
export default defineConfig({
  plugins: [react()],
  base: "/static/",
  build: {
    outDir: "../chat_app/static",
    emptyOutDir: true,
    assetsDir: "assets",
    sourcemap: true,
  },
  server: {
    // Local dev convenience: proxy API calls to the running backend so `npm run
    // dev` works against it without CORS. Production is same-origin (served by
    // the backend), so these only matter in dev.
    proxy: {
      "/chats": "http://127.0.0.1:8000",
      "/runs": "http://127.0.0.1:8000",
      "/artifacts": "http://127.0.0.1:8000",
      "/events": "http://127.0.0.1:8000",
      "/health": "http://127.0.0.1:8000",
      "/lambda": "http://127.0.0.1:8000",
      "/spec-chats": "http://127.0.0.1:8000",
    },
  },
});
