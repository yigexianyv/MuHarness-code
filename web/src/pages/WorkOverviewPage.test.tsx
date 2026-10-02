import { QueryClient, QueryClientProvider } from '@tanstack/react-query'
import { renderToStaticMarkup } from 'react-dom/server'
import { describe, expect, it } from 'vitest'

import WorkOverviewPage from './WorkOverviewPage'

function renderOverview(waitingForRuns: boolean, inboxPending: boolean): string {
  const client = new QueryClient({ defaultOptions: { queries: { retry: false } } })
  client.setQueryData(['conversations'], [{ id: 'c1', title: '正在执行的工作', message_count: 2,
    created_at: '2026-09-30T00:00:00Z', updated_at: '2026-09-30T00:00:00Z' }])
  if (!waitingForRuns) client.setQueryData(['runs'], [])
  return renderToStaticMarkup(<QueryClientProvider client={client}>
    <WorkOverviewPage connected inboxPending={inboxPending} inboxItems={[]} meas={[]}
      onOpenConversation={() => undefined} onOpenRun={() => undefined} onNewWork={() => undefined}
      onNavigate={() => undefined} onOpenArtifacts={() => undefined} />
  </QueryClientProvider>)
}

describe('work overview initial loading', () => {
  it('does not classify a conversation before its execution records arrive', () => {
    const markup = renderOverview(true, false)
    expect(markup).toContain('正在汇总工作与决定')
    expect(markup).not.toContain('预览 正在执行的工作')
  })

  it('waits for decisions before placing work into a lane', () => {
    const markup = renderOverview(false, true)
    expect(markup).toContain('正在汇总工作与决定')
    expect(markup).not.toContain('预览 正在执行的工作')
  })
})
