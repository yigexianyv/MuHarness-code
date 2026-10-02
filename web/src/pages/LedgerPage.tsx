import { useQuery } from '@tanstack/react-query'
import { useMemo } from 'react'

import { mergeMeas } from '../agent/meaPresentation'
import { listAllMeas } from '../api/mea'
import type { MeaRun } from '../api/types'
import { Icon } from '../components/Icon'
import { PageShell } from '../components/PageShell'
import { useEventsStore } from '../stores/events'
import ArtifactsSection from './ArtifactsPage'
import RunsLedger from './RunsPage'

export type LedgerTab = 'runs' | 'artifacts'

/**
 * 账本：执行记录和交付物放在一处。
 * 长任务的每一步都经过只读审计者独立验收，账本上直接给出结论；
 * 普通执行没有独立审计，标成“未经审计”，两类结果不混在一起算。
 */
export default function LedgerPage({
  openRun,
  tab,
  onTabChange,
}: {
  openRun: (runId: string) => void
  tab: LedgerTab
  onTabChange: (tab: LedgerTab) => void
}): React.JSX.Element {
  const measQuery = useQuery({
    queryKey: ['inbox', 'meas'],
    queryFn: () => listAllMeas({ limit: 100 }),
    refetchInterval: 5000,
  })
  const pushed = useEventsStore((state) => state.meaById)
  const meas = useMemo(() => {
    const byId = new Map<string, MeaRun>()
    for (const mea of mergeMeas(measQuery.data ?? [], Object.values(pushed))) byId.set(mea.id, mea)
    return byId
  }, [measQuery.data, pushed])

  return (
    <PageShell
      title="账本"
      subtitle="从执行过程追溯到最终交付，查看每一次工作的真实结果。"
      maxWidth={1360}
      actions={
        <div className="segmented-control ledger-tabs" role="tablist" aria-label="账本分页">
          {(['runs', 'artifacts'] as const).map((item) => (
            <button
              key={item}
              type="button"
              role="tab"
              aria-selected={tab === item}
              className={tab === item ? 'active' : ''}
              onClick={() => onTabChange(item)}
            >
              {item === 'runs' ? '执行记录' : '交付物'}
            </button>
          ))}
        </div>
      }
    >
      <div className="page-inline-note ledger-guide">
        <Icon name={tab === 'runs' ? 'runs' : 'artifacts'} size={18} />
        <div>
          <strong>{tab === 'runs' ? '执行与审计记录' : '已发布的交付物'}</strong>
          <p>{tab === 'runs'
            ? '打开一条记录，检查工具调用、用量与终止原因。长任务独立审计的结论单独标注。'
            : '这里收录通过发布工具保存的文件。查看来源与校验信息，再下载需要的结果。'}</p>
        </div>
      </div>
      {tab === 'runs' ? <RunsLedger openRun={openRun} meas={meas} /> : <ArtifactsSection />}
    </PageShell>
  )
}
