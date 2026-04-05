/* Backend API client for NexusScholar */

import type { ChatRequest, SSEEvent, Conversation, PaperResult } from '../types';

const BASE = '/api';

/** Stream chat via SSE. Yields parsed { event, data } objects. */
export async function* streamChat(req: ChatRequest): AsyncGenerator<SSEEvent> {
    const res = await fetch(`${BASE}/chat`, {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify(req),
    });
    if (!res.ok) throw new Error(`Chat failed: ${res.status}`);

    const reader = res.body!.getReader();
    const decoder = new TextDecoder();
    let buffer = '';

    while (true) {
        const { done, value } = await reader.read();
        if (done) break;
        buffer += decoder.decode(value, { stream: true });
        const lines = buffer.split('\n');
        buffer = lines.pop() || '';

        let currentEvent = '';
        for (const line of lines) {
            if (line.startsWith('event: ')) {
                currentEvent = line.slice(7).trim();
            } else if (line.startsWith('data: ') && currentEvent) {
                try {
                    yield { event: currentEvent, data: JSON.parse(line.slice(6)) };
                } catch { /* skip malformed */ }
            }
        }
    }
}

/** Upload a PDF for ingestion. */
export async function uploadPDF(file: File) {
    const form = new FormData();
    form.append('file', file);
    const res = await fetch(`${BASE}/ingest`, { method: 'POST', body: form });
    if (!res.ok) throw new Error(`Upload failed: ${res.status}`);
    return res.json();
}

/** Get evidence table for an answer. */
export async function getEvidence(answerId: string) {
    const res = await fetch(`${BASE}/evidence/${answerId}`);
    if (!res.ok) throw new Error(`Evidence fetch failed: ${res.status}`);
    return res.json();
}

/** Search papers in the corpus. */
export async function searchPapers(q: string, opts: Record<string, string> = {}) {
    const params = new URLSearchParams({ q, ...opts });
    const res = await fetch(`${BASE}/papers/search?${params}`);
    if (!res.ok) throw new Error(`Search failed: ${res.status}`);
    return res.json();
}

/** List conversations. */
export async function getConversations(): Promise<Conversation[]> {
    const res = await fetch(`${BASE}/conversations`);
    if (!res.ok) return [];
    return res.json();
}

/** Get messages for a conversation. */
export async function getMessages(conversationId: string) {
    const res = await fetch(`${BASE}/conversations/${conversationId}/messages`);
    if (!res.ok) return [];
    return res.json();
}

/** Rebuild all indexes. */
export async function rebuildIndexes() {
    const res = await fetch(`${BASE}/indexes/rebuild`, { method: 'POST' });
    return res.json();
}