/* Zustand store for evidence panel state */

import { create } from 'zustand';

interface EvidenceState {
    isOpen: boolean;
    activeTab: 'evidence' | 'discovery';
    expandedCardId: string | null;
    highlightedCitation: string | null;

    toggle: () => void;
    open: () => void;
    close: () => void;
    setTab: (tab: 'evidence' | 'discovery') => void;
    toggleCard: (id: string) => void;
    highlightCitation: (num: string | null) => void;
}

export const useEvidenceStore = create<EvidenceState>((set) => ({
    isOpen: false,
    activeTab: 'evidence',
    expandedCardId: null,
    highlightedCitation: null,

    toggle: () => set(s => ({ isOpen: !s.isOpen })),
    open: () => set({ isOpen: true }),
    close: () => set({ isOpen: false }),
    setTab: (tab) => set({ activeTab: tab }),
    toggleCard: (id) => set(s => ({ expandedCardId: s.expandedCardId === id ? null : id })),
    highlightCitation: (num) => {
        set({ highlightedCitation: num });
        if (num) setTimeout(() => set({ highlightedCitation: null }), 3000);
    },
}));