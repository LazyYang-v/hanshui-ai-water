import { createApp } from 'vue'
import { createRouter, createWebHistory } from 'vue-router'
import App from './App.vue'
import './style.css'

const router = createRouter({
  history: createWebHistory(),
  routes: [
    { path: '/', component: () => import('./views/Overview.vue'), meta: { title: '集控总览' } },
    { path: '/aeration', component: () => import('./views/Aeration.vue'), meta: { title: '智能曝气' } },
    { path: '/plants', redirect: '/plants/P01' },
    { path: '/plants/:code', component: () => import('./views/PlantDetail.vue'), meta: { title: '厂站详情' } },
    { path: '/alarms', component: () => import('./views/Alarms.vue'), meta: { title: '告警工单' } },
    { path: '/vision', component: () => import('./views/Vision.vue'), meta: { title: '视觉巡检' } },
    { path: '/reports', component: () => import('./views/Reports.vue'), meta: { title: '报告中心' } },
    { path: '/demo', component: () => import('./views/Demo.vue'), meta: { title: '演示控制' } },
  ],
})

createApp(App).use(router).mount('#app')
