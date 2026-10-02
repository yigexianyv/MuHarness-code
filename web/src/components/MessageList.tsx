import { memo } from 'react'
import type { Message } from '../api/types'
import { EmptyState } from './ui'
import { AssistantContent } from './AssistantContent'

export { AssistantContent }



interface RenderedTurn {
  key: number
  role: 'user' | 'assistant'

  author: boolean
  content: string
}

function buildThread(messages: Message[]): RenderedTurn[] {
  const out: RenderedTurn[] = []
  let assistantOpen = false

  messages.forEach((message, index) => {
    switch (message.role) {
      case 'system':

        return
      case 'tool':

        return
      case 'user': {
        assistantOpen = false
        out.push({
          key: index,
          role: 'user',
          author: true,
          content: message.content ?? '',
        })
        return
      }
      case 'assistant': {


        if (message.tool_calls && message.tool_calls.length > 0) {
          return
        }


        const content = message.content ?? ''
        if (!content) return

        out.push({
          key: index,
          role: 'assistant',
          author: !assistantOpen,
          content,
        })
        assistantOpen = true
        return
      }
      default:

        return
    }
  })

  return out
}


export default memo(function MessageList({
  messages,
}: {
  messages: Message[]
}): React.JSX.Element {
  if (messages.length === 0) {
    return (
      <EmptyState
        title="开始对话"
        hint="向 MuHarness 描述你想做的事，Enter 发送。"
      />
    )
  }
  const thread = buildThread(messages)
  return (
    <div>
      {thread.map((turn) => {
        if (turn.role === 'user') {
          return (
            <div key={turn.key} className="message-user">
              <div className="message-user__body">{turn.content}</div>
            </div>
          )
        }
        if (!turn.author) {

          return (
            <div
              key={turn.key}
              className="message-assistant message-assistant--continuation"
            >
              <AssistantContent content={turn.content} />
            </div>
          )
        }
        return (
          <div key={turn.key} className="message-assistant">
            <div className="message-assistant__author">
              <span className="message-assistant__avatar" aria-hidden="true" />
              MuHarness
            </div>
            <AssistantContent content={turn.content} />
          </div>
        )
      })}
    </div>
  )
})
