import { useQuery } from '@tanstack/react-query'
import { useState } from 'react'

import { getSystemInfo } from '../api/system'
import ExtensionsSettings from '../components/ExtensionsSettings'
import { Icon } from '../components/Icon'
import ModelSettingsPanel from '../components/ModelSettingsPanel'
import { ErrorState } from '../components/PageStates'
import { PageShell } from '../components/PageShell'
import { Button } from '../components/ui'

export default function SettingsPage(): React.JSX.Element {
  const [section, setSection] = useState<'general' | 'models' | 'extensions'>('general')

  const infoQuery = useQuery({
    queryKey: ['system-info'],
    queryFn: () => getSystemInfo(),
    refetchInterval: 5000,
    retry: false,
  })

  return (
    <PageShell
      title="设置"
      subtitle="连接模型，配置工作能力，让本地工作台按你的方式运行。"
      maxWidth={1120}
      actions={section === 'general' ? (
        <Button size="sm" variant="ghost" disabled={infoQuery.isFetching} onClick={() => void infoQuery.refetch()}>
          <Icon name="activity" size={14} />检查连接
        </Button>
      ) : undefined}
    >
      <div className="settings-layout">
        <aside className="settings-nav" aria-label="设置分类">
          <button type="button" aria-pressed={section === 'general'} className={section === 'general' ? 'active' : ''} onClick={() => setSection('general')}>
            <Icon name="activity" size={18} className="settings-nav__icon" />
            <div className="settings-nav__text"><strong>通用</strong><span>连接与本地环境</span></div>
          </button>
          <button type="button" aria-pressed={section === 'models'} className={section === 'models' ? 'active' : ''} onClick={() => setSection('models')}>
            <Icon name="agent" size={18} className="settings-nav__icon" />
            <div className="settings-nav__text"><strong>模型</strong><span>对话与后台任务</span></div>
          </button>
          <button type="button" aria-pressed={section === 'extensions'} className={section === 'extensions' ? 'active' : ''} onClick={() => setSection('extensions')}>
            <Icon name="automations" size={18} className="settings-nav__icon" />
            <div className="settings-nav__text"><strong>扩展能力</strong><span>Skills 与 MCP</span></div>
          </button>
        </aside>

        <main className="settings-content">
          {section === 'extensions' ? <ExtensionsSettings /> : section === 'models' ? <ModelSettingsPanel /> : (
            <div className="settings-general">
              <header className="settings-content__header">
                <span className="settings-content__eyebrow">本地工作环境</span>
                <h2>连接与运行状态</h2>
                <p>工作台通过 MuHarness Host 执行任务，这里的信息来自当前连接。</p>
              </header>

              <section className="settings-group">
                <header className="settings-group__header">
                  <div><h3>MuHarness Host</h3><p>当前服务、模型与数据位置</p></div>
                  {!infoQuery.isLoading && !infoQuery.isError ? <span className="settings-status"><i />已连接</span> : null}
                </header>
                {infoQuery.isLoading ? (
                  <div className="settings-loading"><span className="spinner" />正在检查 Host…</div>
                ) : infoQuery.isError ? (
                  <ErrorState
                    message="无法连接 MuHarness Host"
                    hint="请在 backend 目录运行 python -m app.server，然后重试。"
                    onRetry={() => void infoQuery.refetch()}
                  />
                ) : (
                  <dl className="settings-info-list">
                    <InfoRow label="模型提供商" value={infoQuery.data?.provider ?? '—'} />
                    <InfoRow label="当前模型" value={infoQuery.data?.model ?? '—'} />
                    <InfoRow label="Host 版本" value={infoQuery.data?.version ?? '—'} />
                    <InfoRow label="数据库" value={infoQuery.data?.database ?? '—'} mono />
                  </dl>
                )}
              </section>

              <section className="settings-group">
                <header className="settings-group__header">
                  <div><h3>Web 客户端</h3><p>浏览器界面版本</p></div>
                </header>
                <dl className="settings-info-list">
                  <InfoRow label="应用版本" value="0.1.0" />
                  <InfoRow label="运行方式" value="Browser" />
                </dl>
              </section>
            </div>
          )}
        </main>
      </div>
    </PageShell>
  )
}

function InfoRow({
  label,
  value,
  mono = false,
}: {
  label: string
  value: string
  mono?: boolean
}): React.JSX.Element {
  return <div><dt>{label}</dt><dd className={mono ? 'mono' : undefined}>{value}</dd></div>
}
