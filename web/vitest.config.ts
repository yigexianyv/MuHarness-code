import { defineConfig, mergeConfig } from 'vitest/config'

import appConfig from './vite.config.mts'

export default mergeConfig(appConfig, defineConfig({
  test: {
    environment: 'node',
    include: ['src/**/*.test.{ts,tsx}'],
    clearMocks: true,
    restoreMocks: true,
    unstubGlobals: true,
    unstubEnvs: true,
  },
}))
