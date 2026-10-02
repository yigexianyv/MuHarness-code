

import type { ReactElement } from 'react'

import { Icon, type IconName } from './Icon'

export const EXAMPLE_PROMPTS = [
  '分析当前项目并生成一份改进报告',
  '整理一个文件夹里的文件并生成清单',
  '创建一个每天早晨执行的信息简报',
  '调研一个主题并制定可执行的计划',
] as const

const STARTERS: ReadonlyArray<{
  title: string
  description: string
  prompt: (typeof EXAMPLE_PROMPTS)[number]
  icon: IconName
}> = [
  {
    title: '分析项目',
    description: '看清代码结构，找出问题与下一步',
    prompt: EXAMPLE_PROMPTS[0],
    icon: 'agent',
  },
  {
    title: '整理文件',
    description: '归类现有文件，生成可核对的清单',
    prompt: EXAMPLE_PROMPTS[1],
    icon: 'file',
  },
  {
    title: '安排自动化',
    description: '设定时间，让重复工作按计划运行',
    prompt: EXAMPLE_PROMPTS[2],
    icon: 'automations',
  },
  {
    title: '调研与规划',
    description: '收集依据，把目标拆成可执行步骤',
    prompt: EXAMPLE_PROMPTS[3],
    icon: 'runs',
  },
]

export default function ChatEmptyState({
  onSelectPrompt,
}: {
  onSelectPrompt: (prompt: string) => void
}): ReactElement {
  return (
    <section className="chat-empty" aria-label="开始新会话">
      <div className="chat-empty__lead">
        <div className="chat-empty__eyebrow">
          <span className="chat-empty__mark" aria-hidden="true"><Icon name="agent" size={22} /></span>
          <span className="studio-kicker">任务启动台</span>
        </div>
        <h1>从一个清晰的<br /><span>目标开始。</span></h1>
        <p className="chat-empty__intro">
          说明你要完成什么、可以使用哪些资料，以及怎样算完成。
        </p>
      </div>
      <div className="chat-empty__section-label">选一个起点，或直接写下你的任务</div>
      <div className="chat-empty__prompts">
        {STARTERS.map((starter, index) => (
          <button
            key={starter.prompt}
            type="button"
            aria-label={`使用示例：${starter.prompt}`}
            onClick={() => onSelectPrompt(starter.prompt)}
          >
            <span className="starter-number" aria-hidden="true">0{index + 1}</span>
            <span className="chat-empty__prompt-icon">
              <Icon name={starter.icon} size={17} />
            </span>
            <span className="chat-empty__prompt-copy">
              <strong>{starter.title}</strong>
              <small>{starter.description}</small>
            </span>
            <Icon name="plus" size={15} className="chat-empty__prompt-action" />
          </button>
        ))}
      </div>
      <div className="chat-empty__steps" aria-label="任务输入建议">
        <span><strong>01</strong> 写目标</span>
        <span><strong>02</strong> 给资料与约束</span>
        <span><strong>03</strong> 说清验收标准</span>
      </div>
    </section>
  )
}
