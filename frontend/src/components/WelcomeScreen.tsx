import React from 'react';
import { Search, BarChart2, BookOpen, TrendingUp, Database, GitBranch } from 'lucide-react';

const STARTERS = [
    {
        icon: BookOpen,
        label: 'Literature Survey',
        text: 'What are the recent advances in retrieval-augmented generation for scientific QA?',
    },
    {
        icon: BarChart2,
        label: 'Benchmark Comparison',
        text: 'Compare BERT, RoBERTa, and DeBERTa performance on the GLUE benchmark',
    },
    {
        icon: Search,
        label: 'Method Explanation',
        text: 'How does the mixture-of-experts architecture work in Mixtral?',
    },
    {
        icon: TrendingUp,
        label: 'Trend Analysis',
        text: 'What is the trend in parameter-efficient fine-tuning methods since 2022?',
    },
    {
        icon: Database,
        label: 'Dataset Discovery',
        text: 'Which datasets are commonly used for evaluating biomedical NER models?',
    },
    {
        icon: GitBranch,
        label: 'Contradiction Check',
        text: 'Is there conflicting evidence on whether scaling laws hold for vision transformers?',
    },
];

interface Props {
    onSend: (query: string) => void;
}

export default function WelcomeScreen({ onSend }: Props) {
    return (
        <div className="welcome-state">
            <div className="welcome-logo-wrap">
                <div className="welcome-logo">
                    <svg width="30" height="30" viewBox="0 0 24 24" fill="none"
                        stroke="currentColor" strokeWidth="1.7"
                        strokeLinecap="round" strokeLinejoin="round">
                        <circle cx="12" cy="12" r="3" />
                        <path d="M12 2v4M12 18v4M4.93 4.93l2.83 2.83M16.24 16.24l2.83 2.83
                                 M2 12h4M18 12h4M4.93 19.07l2.83-2.83M16.24 7.76l2.83-2.83" />
                    </svg>
                </div>
            </div>

            <h1 className="welcome-title">NexusScholar</h1>
            <p className="welcome-subtitle">
                Enterprise research intelligence — every claim grounded in indexed scientific
                evidence with traceable citations. Upload PDFs or query the corpus directly.
            </p>

            <div className="starter-grid">
                {STARTERS.map((s, i) => {
                    const Icon = s.icon;
                    return (
                        <button key={i} className="starter-card" onClick={() => onSend(s.text)}>
                            <div className="starter-card-label">
                                <Icon size={9} style={{ marginRight: 4, verticalAlign: 'middle' }} />
                                {s.label}
                            </div>
                            <div className="starter-card-text">{s.text}</div>
                        </button>
                    );
                })}
            </div>
        </div>
    );
}
