import { isValidElement, type ReactElement, type ReactNode } from 'react'
import { renderToStaticMarkup } from 'react-dom/server'
import { afterEach, beforeEach, describe, expect, it, vi } from 'vitest'

import type { Artifact } from '../api/artifacts'
import { SERVER_URL } from '../api/config'
import { ArtifactsView } from '../pages/ArtifactsPage'
import { RunArtifactsSection } from '../pages/RunDetailPage'
import ArtifactList, { formatSize } from './ArtifactList'

const CLOCK = new Date(2026, 9, 1, 12)
const DAY_BEFORE = new Date(2026, 8, 30, 12)
const contentUrl = (id: string): string => SERVER_URL + '/artifacts/' + encodeURIComponent(id) + '/content'

function fileResult(overrides: Partial<Artifact> = {}): Artifact {
  return {
    id: 'delivery-file',
    kind: 'file',
    title: '工作复盘',
    description: '本轮工作的最终交付',
    filename: 'review.md',
    mime_type: 'text/markdown',
    size_bytes: 2048,
    sha256: '0123456789abcdef'.repeat(4),
    run_id: 'run-123456789',
    conversation_id: 'chat-987654321',
    task_id: null,
    source_url: null,
    created_at: CLOCK.toISOString(),
    ...overrides,
  }
}

function linkResult(overrides: Partial<Artifact> = {}): Artifact {
  return fileResult({
    id: 'delivery-link',
    kind: 'url',
    filename: null,
    mime_type: null,
    size_bytes: 0,
    sha256: null,
    source_url: 'https://example.com/review',
    ...overrides,
  })
}

function renderList(artifacts: Artifact[], compact = false): string {
  return renderToStaticMarkup(<ArtifactList artifacts={artifacts} compact={compact} />)
}

function visibleText(html: string): string {
  return html.replace(/<[^>]+>/g, '')
}

function entryTitles(html: string): string[] {
  return Array.from(html.matchAll(/<article\b[^>]*aria-label="([^"]+)"/g), (match) => match[1])
}

interface ElementProps {
  children?: ReactNode
  onClick?: () => void
}

// Node 环境没有 DOM；展开无状态组件，直接执行真实按钮处理器。
function findButtons(node: ReactNode): ReactElement<ElementProps>[] {
  if (Array.isArray(node)) return node.flatMap(findButtons)
  if (!isValidElement<ElementProps>(node)) return []
  if (node.type === 'button') return [node]
  if (typeof node.type === 'function') {
    const component = node.type as (props: ElementProps) => ReactNode
    return findButtons(component(node.props))
  }
  return findButtons(node.props.children)
}

beforeEach(() => {
  vi.useFakeTimers({ toFake: ['Date'] })
  vi.setSystemTime(CLOCK)
})

afterEach(() => {
  vi.useRealTimers()
})

describe('交付清单：数据与阅读顺序', () => {
  it.each([false, true])('空数据在 compact=%s 时给出可访问的空提示', (compact) => {
    const html = renderList([], compact)
    expect(html).toContain('role="status"')
    expect(visibleText(html)).toContain('暂无交付结果。')
    expect(entryTitles(html)).toEqual([])
    expect(html).not.toContain('<h2>')
  })

  it('概览按真实数据计算文件、链接和总数', () => {
    const html = renderList([fileResult(), linkResult(), fileResult({ id: 'second-file' })])
    const overview = html.match(/<header\b[\s\S]*?<\/header>/)?.[0] ?? ''
    expect(overview).toContain('aria-label="交付概览"')
    expect(visibleText(overview)).toContain('全部结果3件文件2链接1')
    expect(entryTitles(html)).toHaveLength(3)
  })

  it('交错日期按首次出现分组，同一天的条目保留输入顺序', () => {
    const artifacts = [
      fileResult({ id: 'older-first', title: '先收到的昨日文件', created_at: DAY_BEFORE.toISOString() }),
      fileResult({ id: 'today', title: '今日文件' }),
      linkResult({ id: 'older-last', title: '后收到的昨日链接', created_at: new Date(2026, 8, 30, 18).toISOString() }),
    ]
    const original = structuredClone(artifacts)
    artifacts.forEach(Object.freeze)
    Object.freeze(artifacts)

    const html = renderList(artifacts)
    const dates = Array.from(html.matchAll(/<h2>(.*?)<\/h2>/g), (match) => match[1])
    expect(dates).toEqual([
      DAY_BEFORE.toLocaleDateString('zh-CN', { year: 'numeric', month: 'long', day: 'numeric' }),
      '今天',
    ])
    expect(entryTitles(html)).toEqual(['先收到的昨日文件', '后收到的昨日链接', '今日文件'])
    expect(visibleText(html)).toContain('2 件交付')
    expect(artifacts).toEqual(original)
  })

  it('同月同日但不同年份不会合并为一组', () => {
    const html = renderList([
      fileResult({ id: 'last-year', created_at: new Date(2025, 9, 1, 12).toISOString() }),
      fileResult(),
    ])
    expect(html.match(/<section\b/g)).toHaveLength(2)
    expect(visibleText(html)).toContain('2025年10月1日')
    expect(visibleText(html)).toContain('今天')
  })

  it('无效日期统一归入“更早”，交付物仍可打开', () => {
    const html = renderList([
      fileResult({ created_at: 'invalid-date' }),
      linkResult({ created_at: '' }),
    ])
    expect(html.match(/<h2>更早<\/h2>/g)).toHaveLength(1)
    expect(entryTitles(html)).toHaveLength(2)
    expect(html).toContain('href="' + contentUrl('delivery-file') + '"')
    expect(html).not.toContain('Invalid Date')
  })

  it.each([false, true])('compact=%s 保留描述、文件名、类型和大小', (compact) => {
    const text = visibleText(renderList([fileResult()], compact))
    for (const detail of ['工作复盘', '本轮工作的最终交付', 'review.md', 'text/markdown', '2.0 KB']) {
      expect(text).toContain(detail)
    }
  })

  it.each([
    { title: '', filename: 'fallback.txt', expected: 'fallback.txt' },
    { title: '', filename: null, expected: '交付物' },
  ])('标题缺失时回退到 $expected', ({ title, filename, expected }) => {
    expect(entryTitles(renderList([fileResult({ title, filename, description: null })]))).toEqual([expected])
  })

  it('用户提供的文本按文本转义，不能注入标签或事件', () => {
    const html = renderList([fileResult({ title: '<script>alert(1)</script>', description: '<img src=x onerror=alert(1)>' })])
    expect(html).toContain('&lt;script&gt;alert(1)&lt;/script&gt;')
    expect(html).toContain('&lt;img src=x onerror=alert(1)&gt;')
    expect(html).not.toContain('<script>')
    expect(html).not.toContain('<img ')
  })
})

describe('交付清单：操作契约', () => {
  it.each([false, true])('compact=%s 的文件打开与下载使用编码后的 ID', (compact) => {
    const artifact = fileResult({ id: 'report/review?version=2' })
    const html = renderList([artifact], compact)
    const hrefs = Array.from(html.matchAll(/\bhref="([^"]+)"/g), (match) => match[1])
    expect(hrefs).toEqual([contentUrl(artifact.id), contentUrl(artifact.id)])
    expect(html).toContain('target="_blank"')
    expect(html).toContain('rel="noopener noreferrer"')
    expect(html).toContain('download="review.md"')
    expect(visibleText(html)).toContain('打开')
    expect(visibleText(html)).toContain('下载')
  })

  it('文件名缺失时下载入口不写入虚假的文件名', () => {
    const html = renderList([fileResult({ filename: null })])
    expect(html).toContain('href="' + contentUrl('delivery-file') + '"')
    expect(html).not.toMatch(/\bdownload=/)
    expect(visibleText(html)).toContain('下载')
  })

  it.each([false, true])('compact=%s 的链接使用按钮，没有文件下载或原生 URL 锚点', (compact) => {
    const html = renderList([linkResult()], compact)
    expect(visibleText(html)).toContain('https://example.com/review')
    expect(visibleText(html)).toContain('打开链接')
    expect(html).toContain('<button type="button"')
    expect(html).not.toMatch(/<a\b[^>]*href=/)
    expect(html).not.toContain('下载')
  })

  it.each([false, true])('compact=%s 点击链接以隔离窗口打开真实来源', (compact) => {
    const open = vi.fn()
    vi.stubGlobal('window', { open })
    const buttons = findButtons(ArtifactList({ artifacts: [linkResult()], compact }))
    expect(buttons).toHaveLength(1)
    buttons[0].props.onClick?.()
    expect(open).toHaveBeenCalledTimes(1)
    expect(open).toHaveBeenCalledWith('https://example.com/review', '_blank', 'noopener,noreferrer')
  })

  it.each([null, ''])('链接来源为 %s 时点击不会打开空窗口', (source_url) => {
    const open = vi.fn()
    vi.stubGlobal('window', { open })
    const buttons = findButtons(ArtifactList({ artifacts: [linkResult({ source_url })] }))
    expect(buttons).toHaveLength(1)
    buttons[0].props.onClick?.()
    expect(open).not.toHaveBeenCalled()
  })
})

describe('交付清单：来源与紧凑嵌入', () => {
  it.each([
    { run_id: 'run-123456789', conversation_id: 'chat-987654321', expected: ['执行 run-1234', '会话 chat-987'] },
    { run_id: null, conversation_id: 'chat-987654321', expected: ['执行记录不可用', '会话 chat-987'] },
    { run_id: 'run-123456789', conversation_id: null, expected: ['执行 run-1234'] },
    { run_id: null, conversation_id: null, expected: ['执行记录不可用'] },
  ])('普通模式准确展示来源 $run_id / $conversation_id', ({ run_id, conversation_id, expected }) => {
    const html = renderList([fileResult({ run_id, conversation_id })])
    const text = visibleText(html)
    expect(html).toContain('aria-label="交付来源"')
    expected.forEach((label) => expect(text).toContain(label))
    if (!conversation_id) expect(text).not.toContain('会话 ')
    expect(text).not.toContain('null')
  })

  it('紧凑模式维持原始交错顺序，不展示概览、日期和来源', () => {
    const html = renderList([
      fileResult({ id: 'old-first', title: '旧文件一', created_at: DAY_BEFORE.toISOString() }),
      linkResult({ title: '新链接' }),
      fileResult({ id: 'old-last', title: '旧文件二', created_at: DAY_BEFORE.toISOString() }),
    ], true)
    expect(entryTitles(html)).toEqual(['旧文件一', '新链接', '旧文件二'])
    expect(html).not.toContain('<h2>')
    expect(html).not.toContain('aria-label="交付概览"')
    expect(html).not.toContain('aria-label="交付来源"')
    expect(html).not.toContain('执行 run-1234')
  })
})

describe('文件大小：保留导出函数的单位边界', () => {
  it.each([
    [0, '0 B'], [1, '1 B'], [1023, '1023 B'], [1024, '1.0 KB'],
    [1536, '1.5 KB'], [1024 * 1024, '1.0 MB'], [1024 * 1024 * 2.5, '2.5 MB'],
  ])('%s 字节格式化为 %s', (bytes, expected) => {
    expect(formatSize(bytes as number)).toBe(expected)
  })

  it('列表里的零大小继续显示缺省值', () => {
    expect(visibleText(renderList([fileResult({ size_bytes: 0 })]))).toContain('text/markdown-')
  })
})

describe('交付清单：页面接入', () => {
  it.each([
    { name: '已加载的空列表', artifacts: [], pending: false, expected: '暂无交付结果', absent: '正在加载' },
    { name: '正在加载', artifacts: [fileResult()], pending: true, expected: '正在加载交付物…', absent: '工作复盘' },
    { name: '成功加载', artifacts: [fileResult()], pending: false, expected: '工作复盘', absent: '暂无交付结果' },
  ])('$name 保留原页面状态', ({ artifacts, pending, expected, absent }) => {
    const html = renderToStaticMarkup(<ArtifactsView artifacts={artifacts} pending={pending} />)
    expect(visibleText(html)).toContain(expected)
    expect(visibleText(html)).not.toContain(absent)
    if (artifacts.length === 0) expect(visibleText(html)).toContain('MuHarness 创建的文件和链接会显示在这里。')
  })

  it('错误优先于加载和已有数据，继续提供重试入口', () => {
    const html = renderToStaticMarkup(<ArtifactsView artifacts={[fileResult()]} pending error="交付物加载失败" onRetry={vi.fn()} />)
    expect(visibleText(html)).toContain('交付物加载失败')
    expect(visibleText(html)).toContain('Retry')
    expect(visibleText(html)).not.toContain('工作复盘')
    expect(visibleText(html)).not.toContain('正在加载交付物')
  })

  it('运行详情没有交付物时不生成区块', () => {
    expect(renderToStaticMarkup(<RunArtifactsSection artifacts={[]} />)).toBe('')
  })

  it('运行详情嵌入新清单，保留操作且不重复日期与来源', () => {
    const html = renderToStaticMarkup(<RunArtifactsSection artifacts={[fileResult(), linkResult()]} />)
    expect(html).toContain('<h2>交付物</h2>')
    expect(entryTitles(html)).toHaveLength(2)
    expect(visibleText(html)).toContain('下载')
    expect(visibleText(html)).toContain('打开链接')
    expect(html).not.toContain('aria-label="交付概览"')
    expect(html).not.toContain('aria-label="交付来源"')
    expect(html).not.toContain('<h2>今天</h2>')
  })
})
