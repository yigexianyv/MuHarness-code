

import { describe, expect, it } from 'vitest'
import { renderToStaticMarkup } from 'react-dom/server'

import ChatEmptyState, { EXAMPLE_PROMPTS } from './ChatEmptyState'

describe('ChatEmptyState', () => {
  it('渲染任务启动台、四个可点击示例和输入建议', () => {
    const html = renderToStaticMarkup(<ChatEmptyState onSelectPrompt={() => {}} />)
    expect(html).toContain('从一个清晰的')
    expect(html).toContain('目标开始。')
    expect(html).toContain('任务启动台')
    expect(html).toContain('给资料与约束')
    expect(html).toContain('说清验收标准')
    expect(html.match(/<button/g)).toHaveLength(4)
    for (const prompt of EXAMPLE_PROMPTS) expect(html).toContain(prompt)
  })
})
