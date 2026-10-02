import type { ReactElement } from 'react'

import { buildArtifactDownloadUrl, type Artifact } from '../api/artifacts'
import { Icon } from './Icon'
import './ArtifactList.css'

interface ArtifactListProps {
  artifacts: Artifact[]
  compact?: boolean
}

interface DeliveryDay {
  label: string
  items: Artifact[]
}

function collectDeliveryDays(artifacts: Artifact[]): DeliveryDay[] {
  const today = new Date().toDateString()
  const days = new Map<string, DeliveryDay>()
  for (const artifact of artifacts) {
    const created = new Date(artifact.created_at)
    const label = Number.isNaN(created.getTime())
      ? '更早'
      : created.toDateString() === today
        ? '今天'
        : created.toLocaleDateString('zh-CN', { year: 'numeric', month: 'long', day: 'numeric' })
    const day = days.get(label)
    if (day) day.items.push(artifact)
    else days.set(label, { label, items: [artifact] })
  }
  // 日期按首次出现排列；不重排调用方数据，也不改动任何交付物。
  return [...days.values()]
}

function DeliveryActions({ artifact }: { artifact: Artifact }): ReactElement {
  if (artifact.kind === 'file') {
    const href = buildArtifactDownloadUrl(artifact.id)
    return (
      <div className="delivery-entry__actions" role="group" aria-label="文件操作">
        <a className="delivery-action" href={href} target="_blank" rel="noopener noreferrer">
          <Icon name="external" size={14} /> 打开
        </a>
        <a className="delivery-action delivery-action--primary" href={href} download={artifact.filename ?? undefined}>
          <Icon name="download" size={14} /> 下载
        </a>
      </div>
    )
  }
  return (
    <div className="delivery-entry__actions" role="group" aria-label="链接操作">
      <button
        type="button"
        className="delivery-action delivery-action--primary"
        onClick={() => {
          if (artifact.source_url) window.open(artifact.source_url, '_blank', 'noopener,noreferrer')
        }}
      >
        <Icon name="external" size={14} /> 打开链接
      </button>
    </div>
  )
}

function DeliveryEntry({ artifact, position, showOrigin }: {
  artifact: Artifact
  position: number
  showOrigin: boolean
}): ReactElement {
  const title = artifact.title || artifact.filename || '交付物'
  const isFile = artifact.kind === 'file'
  return (
    <li className="delivery-entry" data-kind={artifact.kind}>
      <span className="delivery-entry__number" aria-hidden="true">
        {String(position).padStart(2, '0')}
      </span>
      <article className="delivery-entry__body" aria-label={title}>
        <div className="delivery-entry__content">
          <div className="delivery-entry__heading">
            <span className="delivery-entry__type">
              <Icon name={isFile ? 'file' : 'external'} size={13} />
              {isFile ? '文件' : '链接'}
            </span>
            <strong className="delivery-entry__title">{title}</strong>
          </div>
          <p className="delivery-entry__location mono">
            {artifact.filename || artifact.source_url || artifact.kind}
          </p>
          {artifact.description ? <p className="delivery-entry__description">{artifact.description}</p> : null}
          <ul className="delivery-entry__details" aria-label="交付物信息">
            <li className="mono">{artifact.mime_type || artifact.kind}</li>
            <li>{artifact.size_bytes ? formatSize(artifact.size_bytes) : '-'}</li>
          </ul>
          {showOrigin ? (
            <div className="delivery-entry__origin" aria-label="交付来源">
              <span><Icon name="runs" size={12} />{artifact.run_id ? `执行 ${artifact.run_id.slice(0, 8)}` : '执行记录不可用'}</span>
              {artifact.conversation_id ? (
                <span><Icon name="chat" size={12} />会话 {artifact.conversation_id.slice(0, 8)}</span>
              ) : null}
            </div>
          ) : null}
        </div>
        <DeliveryActions artifact={artifact} />
      </article>
    </li>
  )
}

function DeliveryEntries({ items, compact = false }: {
  items: Artifact[]
  compact?: boolean
}): ReactElement {
  return (
    <ol className="delivery-register__entries" aria-label="交付条目">
      {items.map((artifact, index) => (
        <DeliveryEntry key={artifact.id} artifact={artifact} position={index + 1} showOrigin={!compact} />
      ))}
    </ol>
  )
}

export default function ArtifactList({ artifacts, compact = false }: ArtifactListProps): ReactElement {
  if (artifacts.length === 0) {
    return (
      <div className="delivery-register__empty" role="status">
        <Icon name="artifacts" size={22} />
        <div><strong>暂无交付结果。</strong><p>完成工作的文件与链接会归档在这里。</p></div>
      </div>
    )
  }
  if (compact) {
    return (
      <div className="delivery-register delivery-register--compact">
        <DeliveryEntries items={artifacts} compact />
      </div>
    )
  }
  const fileCount = artifacts.filter((artifact) => artifact.kind === 'file').length
  return (
    <div className="delivery-register">
      <header className="delivery-register__overview" aria-label="交付概览">
        <div><span className="delivery-register__eyebrow">DELIVERY REGISTER</span><p>交付清单</p></div>
        <dl className="delivery-register__totals">
          <div><dt>全部结果</dt><dd>{artifacts.length}<span>件</span></dd></div>
          <div><dt>文件</dt><dd>{fileCount}</dd></div>
          <div><dt>链接</dt><dd>{artifacts.length - fileCount}</dd></div>
        </dl>
      </header>
      {collectDeliveryDays(artifacts).map(({ label, items }) => (
        <section key={label} className="delivery-register__day" aria-label={label}>
          <div className="delivery-register__date">
            <span className="delivery-register__date-mark" aria-hidden="true" />
            <h2>{label}</h2>
            <span>{items.length} 件交付</span>
          </div>
          <DeliveryEntries items={items} />
        </section>
      ))}
    </div>
  )
}

export function formatSize(bytes: number): string {
  if (bytes < 1024) return `${bytes} B`
  if (bytes < 1024 * 1024) return `${(bytes / 1024).toFixed(1)} KB`
  return `${(bytes / (1024 * 1024)).toFixed(1)} MB`
}
