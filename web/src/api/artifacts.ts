

import { rpcClient } from '../rpc'
import { RpcMethods } from '../rpc/methods'
import { SERVER_URL } from './config'

export type ArtifactKind = 'file' | 'url'

export interface Artifact {
  id: string
  kind: ArtifactKind
  title: string
  description: string | null
  filename: string | null
  mime_type: string | null
  size_bytes: number
  sha256: string | null
  run_id: string | null
  conversation_id: string | null
  task_id: string | null
  source_url: string | null
  created_at: string
}

export interface ArtifactListQuery {
  runId?: string
  conversationId?: string
  limit?: number
}

export async function listArtifacts(
  query: ArtifactListQuery = {},
): Promise<Artifact[]> {
  const params: Record<string, unknown> = { limit: query.limit ?? 50 }
  if (query.runId) params.run_id = query.runId
  if (query.conversationId) params.conversation_id = query.conversationId
  const response = await rpcClient.call<{ artifacts: Artifact[] }>(
    RpcMethods.artifactList,
    params,
  )
  return response.artifacts
}

export async function getArtifact(artifactId: string): Promise<Artifact> {
  const response = await rpcClient.call<{ artifact: Artifact }>(
    RpcMethods.artifactGet,
    { id: artifactId },
  )
  return response.artifact
}


export function buildArtifactDownloadUrl(artifactId: string): string {
  return `${SERVER_URL}/artifacts/${encodeURIComponent(artifactId)}/content`
}
