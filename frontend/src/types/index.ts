/* NexusScholar frontend type definitions */

export type QueryIntent =
    | 'literature_survey' | 'benchmark_comparison' | 'method_explanation'
    | 'paper_lookup' | 'trend_analysis' | 'dataset_discovery'
    | 'author_search' | 'definition' | 'contradiction_check' | 'general';

export type RecencyFilter = 'any' | '1y' | '3y';

export interface ChatRequest {
    query: string;
    conversation_id: string;
    corpus_id?: string;
    recency_filter?: RecencyFilter;
    intent_override?: string;
}

export interface SSEEvent {
    event: string;
    data: Record<string, unknown>;
}

export interface EvidenceRowData {
    evidence_id: string;
    paper_title: string;
    authors: string;
    year?: number;
    venue?: string;
    section: string;
    passage_preview?: string;
    chunk_text?: string;
    relevance: number;
    is_peer_reviewed: boolean;
    is_retracted: boolean;
    source_url?: string;
    pdf_url?: string;
}

export interface EvidenceData {
    answer_id: string;
    total_rows: number;
    confidence: number;
    rows: EvidenceRowData[];
}

export interface CitationCard {
    number: number;
    paper_title: string;
    authors: string;
    year?: number;
    venue?: string;
    passage_preview: string;
    section: string;
    is_peer_reviewed: boolean;
    is_retracted: boolean;
}

export interface EntityWarning {
    entityCorrect: boolean;
    substitutedEntity?: string;
    confidence: number;
    issues: string[];
}

export interface EntityGrounding {
    primarySubject: string;
    entityType: string;
    requiresGrounding: boolean;
}

export interface Message {
    id: string;
    role: 'user' | 'assistant';
    content: string;
    answerId?: string;
    citations?: CitationCard[];
    uncertaintyFlags?: string[];
    totalSources?: number;
    peerReviewedCount?: number;
    preprintCount?: number;
    isAbstention?: boolean;
    isStreaming?: boolean;
    isError?: boolean;
    entityWarning?: EntityWarning;
    entityGrounding?: EntityGrounding;
}

export interface Conversation {
    conversation_id: string;
    title: string;
    corpus_id: string;
    created_at: string;
    updated_at: string;
}

export interface PaperResult {
    paper_id: string;
    title: string;
    authors: string[];
    year?: number;
    venue?: string;
    abstract?: string;
    citation_count?: number;
    is_peer_reviewed: boolean;
}