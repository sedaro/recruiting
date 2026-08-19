import viteReact from '@vitejs/plugin-react';
import { defineConfig } from 'vite';
import viteSvgr from 'vite-plugin-svgr';

// https://vitejs.dev/config/
export default defineConfig({
  base: '/',
  plugins: [viteReact(), viteSvgr()],
  resolve: {
    // Honour the `paths` in tsconfig.json, so `from 'routes'` resolves. Vite 8
    // does this natively; it used to need the vite-tsconfig-paths plugin.
    tsconfigPaths: true,
  },
  server: {
    host: true,
    port: 3030,
  },
});
