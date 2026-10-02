import { useEffect, useState } from 'react'
import { useQueryClient } from '@tanstack/react-query'

import { useEventsStore } from './stores/events'
import DeskNavigation from './components/DeskNavigation'
import type { DeskPage } from './components/DeskNavigation'
import WorkBriefDialog from './components/WorkBriefDialog'
import { ToastViewport } from './components/ToastViewport'
import { Icon } from './components/Icon'
import AutomationsPage from './pages/AutomationsPage'
import ChatPage from './pages/ChatPage'
import InboxPage, { useInbox } from './pages/InboxPage'
import LedgerPage from './pages/LedgerPage'
import type { LedgerTab } from './pages/LedgerPage'
import MemoryPage from './pages/MemoryPage'
import RunDetailPage from './pages/RunDetailPage'
import SettingsPage from './pages/SettingsPage'
import WorkOverviewPage from './pages/WorkOverviewPage'

/**
 * 进行中：chat（工作台）、automations
 * 等你处理：inbox（审批 + 长任务提问/暂停/阻塞 + 可恢复的中断）
 * 已验证：ledger（执行记录 + 审计结论 + 交付物）、memory
 */
export type PageKey =
  | 'chat'
  | 'automations'
  | 'inbox'
  | 'ledger'
  | 'memory'
  | 'settings'

export interface AppState {
  page: PageKey
  selectedRunId: string | null
  navigate: (page: PageKey) => void
  openRun: (runId: string) => void
}

const PAGE_LABELS: Record<DeskPage, string> = {
  overview: '工作总览',
  chat: '工作区',
  automations: '自动化',
  inbox: '决策中心',
  ledger: '成果与记录',
  memory: '记忆',
  settings: '设置',
}

export default function App(): React.JSX.Element {
  const queryClient = useQueryClient()
  const [page, setPage] = useState<DeskPage>('overview')
  const [briefOpen, setBriefOpen] = useState(false)
  const [preparedWork, setPreparedWork] = useState<{
    conversationId: string
    content: string
    mode: 'normal' | 'plan'
  } | null>(null)
  const [selectedRunId, setSelectedRunId] = useState<string | null>(null)
  const [selectedConversationId, setSelectedConversationId] = useState<string | null>(null)
  const [everConnected, setEverConnected] = useState(false)

  const connect = useEventsStore((state) => state.connect)
  const disconnect = useEventsStore((state) => state.disconnect)
  const connected = useEventsStore((state) => state.connected)
  const runStatuses = useEventsStore((state) => state.runStatuses)

  useEffect(() => {
    connect()
    return () => disconnect()
  }, [connect, disconnect])

  useEffect(() => {
    if (connected) setEverConnected(true)
  }, [connected])

  const [ledgerTab, setLedgerTab] = useState<LedgerTab>('runs')
  const inbox = useInbox()

  const runningCount = Object.values(runStatuses).filter(
    (status) => status === 'running',
  ).length

  const navigate = (next: DeskPage): void => {
    // 侧栏点“账本”回到执行记录列表；在账本内看详情时再点一次也回到列表
    if (next === 'ledger') setLedgerTab('artifacts')
    setSelectedRunId(null)
    setPage(next)
  }

  const openRun = (runId: string): void => {
    setSelectedRunId(runId)
    setPage('ledger')
  }

  const openArtifacts = (): void => {
    setSelectedRunId(null)
    setLedgerTab('artifacts')
    setPage('ledger')
  }

  const openConversation = (conversationId: string): void => {
    setSelectedConversationId(conversationId)
    setSelectedRunId(null)
    setPage('chat')
  }

  return (
    <div className={`desk-app desk-app--${page}`}>
      <a className="skip-link" href="#workspace-content">跳到工作区</a>
      <DeskNavigation
        current={page}
        onNavigate={navigate}
        connected={connected}
        inboxCount={inbox.items.length}
        runningCount={runningCount}
        onNewWork={() => setBriefOpen(true)}
      />
      <div className="desk-main" id="workspace-content" tabIndex={-1}>
        {page !== 'overview' ? <header className="desk-location">
          <div className="desk-location__path" aria-label="当前位置">
            <button type="button" onClick={() => navigate('overview')}>工作桌面</button>
            <span aria-hidden="true">/</span>
            <span>{selectedRunId ? '执行详情' : PAGE_LABELS[page]}</span>
          </div>
          <span className="desk-location__hint"><Icon name="activity" size={12} />工作状态来自本地服务</span>
        </header> : null}
        {!connected && everConnected ? (
          <div className="host-banner" role="status">
            <span className="host-banner__dot" />
            Host 连接已断开，正在重连…
          </div>
        ) : null}
        {page === 'overview' && <WorkOverviewPage
          connected={connected}
          inboxItems={inbox.items}
          inboxPending={inbox.pending}
          inboxError={inbox.error}
          onRefreshInbox={inbox.refetch}
          meas={inbox.meas}
          onOpenConversation={openConversation}
          onOpenRun={openRun}
          onOpenArtifacts={openArtifacts}
          onNewWork={() => setBriefOpen(true)}
          onNavigate={navigate}
        />}
        {page === 'chat' && (
          <ChatPage
            onNavigate={(next) => (next === 'ledger' ? openArtifacts() : navigate(next))}
            onOpenRun={openRun}
            initialConversationId={selectedConversationId}
            onConversationChange={setSelectedConversationId}
            initialDraft={preparedWork}
            onDraftConsumed={() => setPreparedWork(null)}
            onNewWork={() => setBriefOpen(true)}
          />
        )}
        {page === 'ledger' &&
          (selectedRunId ? (
            <RunDetailPage
              runId={selectedRunId}
              onBack={() => setSelectedRunId(null)}
              onOpenConversation={openConversation}
            />
          ) : (
            <LedgerPage openRun={openRun} tab={ledgerTab} onTabChange={setLedgerTab} />
          ))}
        {page === 'automations' && <AutomationsPage />}
        {page === 'inbox' && (
          <InboxPage onOpenConversation={openConversation} onOpenRun={openRun} />
        )}
        {page === 'memory' && <MemoryPage />}
        {page === 'settings' && <SettingsPage />}
      </div>
      <WorkBriefDialog
        open={briefOpen}
        onClose={() => setBriefOpen(false)}
        onPrepared={(conversationId, content, mode) => {
          void queryClient.invalidateQueries({ queryKey: ['conversations'] })
          setPreparedWork({ conversationId, content, mode })
          openConversation(conversationId)
          setBriefOpen(false)
        }}
      />
      <ToastViewport />
    </div>
  )
}
