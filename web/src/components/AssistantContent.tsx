






import { Children, isValidElement, memo, useEffect, useRef, useState } from 'react'
import type { ReactElement, ReactNode } from 'react'
import ReactMarkdown from 'react-markdown'
import type { Components } from 'react-markdown'
import remarkGfm from 'remark-gfm'
import { createHighlighterCore } from 'shiki/core'
import type { HighlighterCore } from 'shiki/core'
import { createJavaScriptRegexEngine } from '@shikijs/engine-javascript'

const THEME = 'github-dark'


const LANG_BUNDLES: Record<string, () => Promise<unknown>> = {
  typescript: () => import('@shikijs/langs/typescript'),
  ts: () => import('@shikijs/langs/typescript'),
  javascript: () => import('@shikijs/langs/javascript'),
  js: () => import('@shikijs/langs/javascript'),
  jsx: () => import('@shikijs/langs/jsx'),
  tsx: () => import('@shikijs/langs/tsx'),
  python: () => import('@shikijs/langs/python'),
  py: () => import('@shikijs/langs/python'),
  bash: () => import('@shikijs/langs/bash'),
  shell: () => import('@shikijs/langs/shellscript'),
  sh: () => import('@shikijs/langs/shellscript'),
  shellscript: () => import('@shikijs/langs/shellscript'),
  json: () => import('@shikijs/langs/json'),
  markdown: () => import('@shikijs/langs/markdown'),
  md: () => import('@shikijs/langs/markdown'),
  css: () => import('@shikijs/langs/css'),
  html: () => import('@shikijs/langs/html'),
  yaml: () => import('@shikijs/langs/yaml'),
  sql: () => import('@shikijs/langs/sql'),
  rust: () => import('@shikijs/langs/rust'),
  go: () => import('@shikijs/langs/go'),
}

let highlighterPromise: Promise<HighlighterCore> | null = null

function getHighlighter(): Promise<HighlighterCore> {
  if (!highlighterPromise) {
    const langs = Object.values(LANG_BUNDLES).map((load) => load()) as unknown as Parameters<
      typeof createHighlighterCore
    >[0]['langs']
    highlighterPromise = createHighlighterCore({
      themes: [import('@shikijs/themes/github-dark')],
      langs,
      engine: createJavaScriptRegexEngine(),
    })
  }
  return highlighterPromise
}




function CodeBlock({
  code,
  lang,
  streaming,
}: {
  code: string
  lang: string
  streaming: boolean
}): ReactElement {
  const [html, setHtml] = useState<string | null>(null)
  const timerRef = useRef<number | null>(null)

  useEffect(() => {
    if (streaming) return


    if (timerRef.current !== null) {
      window.clearTimeout(timerRef.current)
    }
    timerRef.current = window.setTimeout(() => {
      ;(async () => {
        try {
          const highlighter = await getHighlighter()
          const out = highlighter.codeToHtml(code, {
            lang,
            theme: THEME,
          })
          setHtml(out)
        } catch {}
      })()
    }, 120)
    return () => {
      if (timerRef.current !== null) {
        window.clearTimeout(timerRef.current)
      }
      timerRef.current = null
    }
  }, [code, lang, streaming])

  if (html !== null) {
    return (
      <div
        className="assistant-code"
        dangerouslySetInnerHTML={{ __html: html }}
      />
    )
  }
  return (
    <pre className="assistant-code">
      <code>{code}</code>
    </pre>
  )
}

const createComponents = (streaming: boolean): Components => ({

  pre({ children }) {
    const child = Children.toArray(children)[0]
    if (isValidElement<{ className?: string; children?: ReactNode }>(child)) {
      const match = /language-(\w+)/.exec(child.props.className ?? '')
      if (match && match[1]) {
        const lang = match[1]
        const text = String(child.props.children ?? '').replace(/\n$/, '')
        if (lang in LANG_BUNDLES) {
          return <CodeBlock code={text} lang={lang} streaming={streaming} />
        }
        return (
          <pre className="assistant-code">
            <code>{text}</code>
          </pre>
        )
      }
    }
    return <pre>{children}</pre>
  },
  code({ className, children }) {
    return <code className={className}>{children}</code>
  },
  a({ href, children }) {
    const external = href?.startsWith('http://') || href?.startsWith('https://')
    return (
      <a
        href={href}
        {...(external ? { target: '_blank', rel: 'noreferrer' } : {})}
      >
        {children}
      </a>
    )
  },
})




export const AssistantContent = memo(function AssistantContent({
  content,
  streaming = false,
}: {
  content: string
  streaming?: boolean
}): React.JSX.Element {
  return (
    <div className="message-assistant__body">
      <ReactMarkdown
        remarkPlugins={[remarkGfm]}
        components={createComponents(streaming)}
      >
        {content}
      </ReactMarkdown>
    </div>
  )
})
