import { defineConfig } from 'vite'
import vue from '@vitejs/plugin-vue'

// 生产环境由 nginx.conf 做同样的代理
const target = process.env.API_TARGET ?? 'http://127.0.0.1:8000'

export default defineConfig({
  plugins: [vue()],
  build: { chunkSizeWarningLimit: 600 },   // ECharts 已按需引入，单独一个约 544 KB 的 chunk，只在用到图表的页面加载
  server: {
    port: 5173,
    proxy: {
      '/api': { target, changeOrigin: true },
      '/ws': { target: target.replace(/^http/, 'ws'), ws: true },
    },
  },
})
