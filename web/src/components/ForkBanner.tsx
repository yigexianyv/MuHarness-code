import { useQueryClient } from '@tanstack/react-query'
import { useState } from 'react'

import { undoRewind } from '../api/runs'
import type { ConversationFork } from '../api/types'
import { toast } from '../stores/toasts'
import { ConfirmDialog } from './ConfirmDialog'

/** 分支会话顶部：说明它从哪里来，可以返回来源或撤销文件回退。 */
export default function ForkBanner({
  fork,
  sourceTitle,
  onOpenSource,
}: {
  fork: ConversationFork
  sourceTitle: string | null
  onOpenSource: (conversationId: string) => void
}): React.JSX.Element {
  const queryClient = useQueryClient()
  const [confirmUndo, setConfirmUndo] = useState(false)
  const [busy, setBusy] = useState(false)

  const undo = async (): Promise<void> => {
    setBusy(true)
    try {
      await undoRewind(fork.rewind_key)
      toast.success('工作区文件已恢复到回退之前')
      void queryClient.invalidateQueries({ queryKey: ['conversation', fork.conversation_id] })
    } catch (cause) {
      toast.error(cause instanceof Error ? cause.message : String(cause))
    } finally {
      setBusy(false)
      setConfirmUndo(false)
    }
  }

  return (
    <div className="fork-banner" role="note">
      <span>
        从《{sourceTitle ?? '原会话'}》第 {fork.source_step} 步之前重新执行
        {fork.undone ? '（文件回退已撤销）' : ''}
      </span>
      <span className="fork-banner__actions">
        <button type="button" className="context-link" onClick={() => onOpenSource(fork.source_conversation_id)}>返回来源</button>
        {!fork.undone ? (
          <button type="button" className="context-link" disabled={busy} onClick={() => setConfirmUndo(true)}>撤销文件回退</button>
        ) : null}
      </span>
      <ConfirmDialog
        open={confirmUndo}
        title="撤销文件回退？"
        message="工作区文件会恢复到点击「重做此步」之前的样子。这个会话和它的消息会保留。"
        confirmLabel="撤销文件回退"
        cancelLabel="取消"
        busy={busy}
        onConfirm={() => void undo()}
        onCancel={() => setConfirmUndo(false)}
      />
    </div>
  )
}
