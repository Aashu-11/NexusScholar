/* Zustand store for chat/conversation state */

import { create } from 'zustand';
import type { Message, Conversation, EvidenceData, EntityGrounding } from '../types';
import { streamChat, getConversations, getMessages, uploadPDF } from '../lib/api';

function uid() {
    return Date.now().toString(36) + Math.random().toString(36).slice(2, 8);
}

interface ChatState {
    conversationId: string;
    conversations: Conversation[];
    messages: Message[];
    isStreaming: boolean;
    pipelineStatus: string | null;
    currentIntent: string | null;
    evidenceData: EvidenceData | null;
    uploadStatus: string | null;
    entityGrounding: EntityGrounding | null;

    newConversation: () => void;
    loadConversations: () => Promise<void>;
    loadConversation: (id: string) => Promise<void>;
    sendMessage: (query: string, recencyFilter?: string) => Promise<void>;
    handleUpload: (file: File) => Promise<void>;
}

export const useChatStore = create<ChatState>((set, get) => ({
    conversationId: uid(),
    conversations: [],
    messages: [],
    isStreaming: false,
    pipelineStatus: null,
    currentIntent: null,
    evidenceData: null,
    uploadStatus: null,
    entityGrounding: null,

    newConversation: () => set({
        conversationId: uid(),
        messages: [],
        evidenceData: null,
        currentIntent: null,
        pipelineStatus: null,
        entityGrounding: null,
    }),

    loadConversations: async () => {
        try {
            const convs = await getConversations();
            set({ conversations: convs });
        } catch { /* ignore */ }
    },

    loadConversation: async (id: string) => {
        set({ conversationId: id });
        try {
            const msgs = await getMessages(id);
            set({
                messages: msgs.map((m: Record<string, string>) => ({
                    id: m.message_id,
                    role: m.role as 'user' | 'assistant',
                    content: m.content,
                    answerId: m.answer_id,
                })),
            });
        } catch { /* ignore */ }
    },

    sendMessage: async (query: string, recencyFilter = 'any') => {
        const { isStreaming, conversationId } = get();
        if (isStreaming || !query.trim()) return;

        const userMsg: Message = { id: uid(), role: 'user', content: query.trim() };
        const assistantMsg: Message = { id: uid(), role: 'assistant', content: '', isStreaming: true };

        set(s => ({
            messages: [...s.messages, userMsg, assistantMsg],
            isStreaming: true,
            pipelineStatus: 'Analyzing query...',
            evidenceData: null,
        }));

        try {
            const stream = streamChat({
                query: query.trim(),
                conversation_id: conversationId,
                recency_filter: recencyFilter as 'any' | '1y' | '3y',
            });

            for await (const { event, data } of stream) {
                switch (event) {
                    case 'intent':
                        if (data.intent) {
                            set({ currentIntent: data.intent as string, pipelineStatus: `Intent: ${(data.intent as string).replace(/_/g, ' ')}` });
                        }
                        break;
                    case 'retrieval':
                        if (data.status === 'searching') set({ pipelineStatus: 'Searching evidence...' });
                        else if (data.status === 'complete' || data.status === 'fused')
                            set({ pipelineStatus: `Found ${data.candidates_found || data.final_candidates} candidates` });
                        break;
                    case 'planning':
                        set({ pipelineStatus: `Planning (confidence: ${Math.round(((data.confidence as number) || 0) * 100)}%)` });
                        break;
                    case 'evidence':
                        set({ evidenceData: data as unknown as EvidenceData, pipelineStatus: `${data.total_rows} sources — synthesizing...` });
                        break;
                    case 'synthesis_token':
                        set(s => {
                            const msgs = [...s.messages];
                            const last = msgs[msgs.length - 1];
                            if (last?.role === 'assistant') {
                                msgs[msgs.length - 1] = { ...last, content: last.content + (data.token as string) };
                            }
                            return { messages: msgs, pipelineStatus: null };
                        });
                        break;
                    case 'math_results':
                        // Sandbox has executed all code blocks — replace the streamed
                        // draft text with the verified version immediately.
                        set(s => {
                            const msgs = [...s.messages];
                            const last = msgs[msgs.length - 1];
                            if (last?.role === 'assistant' && data.updated_text) {
                                msgs[msgs.length - 1] = { ...last, content: data.updated_text as string };
                            }
                            return { messages: msgs, pipelineStatus: 'Math verified ✓' };
                        });
                        break;
                    case 'answer_complete':
                        set(s => {
                            const msgs = [...s.messages];
                            const last = msgs[msgs.length - 1];
                            if (last?.role === 'assistant') {
                                msgs[msgs.length - 1] = {
                                    ...last,
                                    content: data.markdown_text as string,
                                    answerId: data.answer_id as string,
                                    citations: data.citation_cards as Message['citations'],
                                    uncertaintyFlags: data.uncertainty_flags as string[],
                                    totalSources: data.total_sources as number,
                                    peerReviewedCount: data.peer_reviewed_count as number,
                                    preprintCount: data.preprint_count as number,
                                    isAbstention: data.is_abstention as boolean,
                                    isStreaming: false,
                                };
                            }
                            return { messages: msgs };
                        });
                        break;
                    case 'entity_grounding':
                        set({
                            entityGrounding: {
                                primarySubject: data.primary_subject as string,
                                entityType: data.entity_type as string,
                                requiresGrounding: data.requires_grounding as boolean,
                            }
                        });
                        break;
                    case 'entity_warning':
                        set(s => {
                            const msgs = [...s.messages];
                            const last = msgs[msgs.length - 1];
                            if (last?.role === 'assistant') {
                                msgs[msgs.length - 1] = {
                                    ...last,
                                    entityWarning: {
                                        entityCorrect: data.entity_correct as boolean,
                                        substitutedEntity: data.substituted_entity as string,
                                        confidence: data.confidence as number,
                                        issues: data.issues as string[],
                                    }
                                };
                            }
                            return { messages: msgs };
                        });
                        break;
                    case 'error':
                        set(s => {
                            const msgs = [...s.messages];
                            const last = msgs[msgs.length - 1];
                            if (last?.role === 'assistant') {
                                msgs[msgs.length - 1] = { ...last, content: `**Error:** ${data.message}`, isError: true, isStreaming: false };
                            }
                            return { messages: msgs };
                        });
                        break;
                }
            }
        } catch (err) {
            set(s => {
                const msgs = [...s.messages];
                const last = msgs[msgs.length - 1];
                if (last?.role === 'assistant') {
                    msgs[msgs.length - 1] = { ...last, content: `**Connection error:** ${(err as Error).message}`, isError: true, isStreaming: false };
                }
                return { messages: msgs };
            });
        } finally {
            set({ isStreaming: false, pipelineStatus: null });
            get().loadConversations();
        }
    },

    handleUpload: async (file: File) => {
        set({ uploadStatus: `Ingesting ${file.name}...` });
        try {
            const result = await uploadPDF(file);
            set({ uploadStatus: `Ingested: ${file.name} (${result.paper_id})` });
            setTimeout(() => set({ uploadStatus: null }), 4000);
        } catch (err) {
            set({ uploadStatus: `Failed: ${(err as Error).message}` });
            setTimeout(() => set({ uploadStatus: null }), 4000);
        }
    },
}));