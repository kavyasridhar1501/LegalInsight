/**
 * LegalInsight - LangChain.js Engine
 *
 * Loaded as an ES module via <script type="module">.
 * Runs entirely in the browser — no server required, GitHub Pages compatible.
 *
 * Provides:
 *   - RecursiveCharacterTextSplitter  (document chunking)
 *   - ChatPromptTemplate              (structured prompt management)
 *   - Jaccard-based consistency score (better hallucination proxy)
 *   - Structured JSON extraction      (key contract terms)
 *   - Conversation message formatting (follow-up questions)
 *
 * Exposes window.LegalInsightLC for use by the non-module app.js.
 */

import { ChatPromptTemplate } from 'https://esm.sh/@langchain/core@0.3/prompts';
import { HumanMessage, AIMessage } from 'https://esm.sh/@langchain/core@0.3/messages';
import { RecursiveCharacterTextSplitter } from 'https://esm.sh/@langchain/textsplitters@0.1';

// Document Chunker

const textSplitter = new RecursiveCharacterTextSplitter({
    chunkSize: 2000,
    chunkOverlap: 200,
    separators: ['\n\n\n', '\n\n', '\n', '. ', '! ', '? ', '; ', ', ', ' ', '']
});

// Prompt Templates

const LEGAL_SYSTEM = `You are a senior legal analyst specialising in contract law. \
You read contracts carefully and answer questions with precision.

Rules:
- Reference specific clauses or sections when possible
- Distinguish explicit contract terms from general legal implications
- Clearly flag obligations, rights, deadlines, and risks
- If information is absent from the provided text, say: \
  "This is not addressed in the provided contract sections"
- Be direct and concise`;

/** Used for the first analysis of a contract. */
const analysisPrompt = ChatPromptTemplate.fromMessages([
    ['system', LEGAL_SYSTEM],
    ['human', 'Contract sections:\n\n{context}\n\n---\n\nQuestion: {query}']
]);

/** Used when the user asks a follow-up; conversation history is injected manually. */
const followUpSystemMsg = LEGAL_SYSTEM +
    '\n\nYou are continuing an analysis of the same contract. ' +
    'Be consistent with your earlier answers.';

/** Used to extract structured key terms from a contract. */
const extractionPrompt = ChatPromptTemplate.fromMessages([
    [
        'system',
        'You are a legal data-extraction specialist. ' +
        'Extract structured information from the contract below and return ONLY valid JSON — ' +
        'no markdown fences, no explanation, just the JSON object.'
    ],
    [
        'human',
        `Extract and return this JSON structure:
{{
  "contract_type": "string",
  "parties": [{{"name": "string", "role": "string"}}],
  "effective_date": "string or null",
  "term_duration": "string or null",
  "payment_terms": "string or null",
  "termination_notice": "string or null",
  "governing_law": "string or null",
  "auto_renewal": "string or null",
  "liability_cap": "string or null",
  "key_obligations": ["up to 5 strings"],
  "key_risks": ["up to 3 strings"]
}}

Contract:
{contract_text}`
    ]
]);

// Chunk Selection

const STOPWORDS = new Set([
    'what', 'when', 'where', 'who', 'how', 'why', 'which',
    'this', 'that', 'with', 'from', 'have', 'will', 'does',
    'about', 'the', 'and', 'are', 'for', 'not', 'but',
    'can', 'all', 'any', 'its', 'their', 'there', 'been'
]);

function extractQueryKeywords(query) {
    return new Set(
        query.toLowerCase()
            .replace(/[^\w\s]/g, '')
            .split(/\s+/)
            .filter(w => w.length > 3 && !STOPWORDS.has(w))
    );
}

function scoreChunk(text, keywords) {
    const words = text.toLowerCase().replace(/[^\w\s]/g, '').split(/\s+/);
    return words.filter(w => keywords.has(w)).length;
}

/**
 * Splits the contract into chunks with RecursiveCharacterTextSplitter,
 * scores each chunk for relevance to the query, and returns the top chunks
 * joined as a single context string.
 *
 * Short contracts (<3 000 chars) are returned as-is.
 *
 * @param {string} contractText
 * @param {string} query
 * @param {number} maxChunks
 * @returns {Promise<{context: string, used: number, total: number}>}
 */
async function getRelevantChunks(contractText, query, maxChunks = 4) {
    if (contractText.length < 3000) {
        return { context: contractText, used: 1, total: 1 };
    }

    const docs = await textSplitter.createDocuments([contractText]);

    if (docs.length <= maxChunks) {
        return {
            context: docs.map(d => d.pageContent).join('\n\n---\n\n'),
            used: docs.length,
            total: docs.length
        };
    }

    const keywords = extractQueryKeywords(query);

    const scored = docs.map((doc, i) => ({
        content: doc.pageContent,
        index: i,
        score: scoreChunk(doc.pageContent, keywords)
    }));

    // Always keep first chunk (parties, dates) and last chunk (signatures, governing law)
    const first = scored[0];
    const last  = scored[scored.length - 1];
    const middle = scored.slice(1, -1).sort((a, b) => b.score - a.score);

    const slots = maxChunks - 2;
    const selected = [first, ...middle.slice(0, Math.max(0, slots)), last];

    // Restore document order
    selected.sort((a, b) => a.index - b.index);

    return {
        context: selected.map(s => s.content).join('\n\n---\n\n'),
        used: selected.length,
        total: docs.length
    };
}

// Semantic Consistency (Jaccard)

function tokenize(text) {
    return new Set(
        text.toLowerCase()
            .replace(/[^\w\s]/g, '')
            .split(/\s+/)
            .filter(w => w.length > 3)
    );
}

function jaccard(a, b) {
    let inter = 0;
    for (const t of a) { if (b.has(t)) inter++; }
    const union = a.size + b.size - inter;
    return union === 0 ? 1 : inter / union;
}

/**
 * Returns a 0-100 consistency score based on pairwise Jaccard similarity
 * of the response word sets. Much more meaningful than length-variance.
 */
function calculateSemanticConsistency(responses) {
    if (responses.length < 2) return 100;

    const sets = responses.map(tokenize);
    let total = 0, pairs = 0;

    for (let i = 0; i < sets.length; i++) {
        for (let j = i + 1; j < sets.length; j++) {
            total += jaccard(sets[i], sets[j]);
            pairs++;
        }
    }

    const avg = pairs > 0 ? total / pairs : 1;
    return Math.round(Math.max(40, Math.min(100, avg * 100)));
}

// Message Formatting

/** Converts LangChain BaseMessage[] → plain {role, content}[] for provider APIs. */
function lcToPlain(messages) {
    const roleMap = { system: 'system', human: 'user', ai: 'assistant', generic: 'user' };
    return messages.map(m => ({
        role: roleMap[m._getType()] || 'user',
        content: typeof m.content === 'string'
            ? m.content
            : (Array.isArray(m.content) ? m.content.map(c => c.text || '').join('') : '')
    }));
}

/**
 * Builds the messages array for the initial contract analysis.
 * For follow-up questions, history is injected directly.
 *
 * @param {string} context   - relevant contract chunks
 * @param {string} query
 * @param {Array}  history   - [{role, content}, ...] previous exchanges
 * @returns {Promise<Array>} - plain message objects ready for any provider
 */
async function formatAnalysisMessages(context, query, history = []) {
    if (history.length > 0) {
        // Build manually to avoid MessagesPlaceholder complexity
        const historyMessages = history.map(m =>
            m.role === 'user' ? new HumanMessage(m.content) : new AIMessage(m.content)
        );
        const lcMessages = await ChatPromptTemplate.fromMessages([
            ['system', followUpSystemMsg],
            ...historyMessages.map(m => [m._getType() === 'human' ? 'human' : 'ai', m.content]),
            ['human', 'Relevant contract context:\n\n{context}\n\n---\n\nFollow-up: {query}']
        ]).formatMessages({ context, query });

        return lcToPlain(lcMessages);
    }

    const lcMessages = await analysisPrompt.formatMessages({ context, query });
    return lcToPlain(lcMessages);
}

/**
 * Builds the messages array for structured key-terms extraction.
 * Sends only the first portion of the contract to stay within token limits.
 */
async function formatExtractionMessages(contractText) {
    const excerpt = contractText.slice(0, 5000);
    const lcMessages = await extractionPrompt.formatMessages({ contract_text: excerpt });
    return lcToPlain(lcMessages);
}

// Structured Data Parsing & Rendering

function parseStructuredData(text) {
    try {
        const cleaned = text.replace(/```json\n?|```\n?/g, '').trim();
        return JSON.parse(cleaned);
    } catch {
        const match = text.match(/\{[\s\S]*\}/);
        if (match) {
            try { return JSON.parse(match[0]); } catch { /* fall through */ }
        }
        return null;
    }
}

function renderStructuredData(data) {
    if (!data) return null;

    const field = (label, value) =>
        value ? `<div class="lc-field"><span class="lc-label">${label}</span><span class="lc-value">${value}</span></div>` : '';

    const list = (label, items, cls = '') =>
        items && items.length
            ? `<div class="lc-field lc-list-field ${cls}">
                 <span class="lc-label">${label}</span>
                 <ul>${items.map(i => `<li>${i}</li>`).join('')}</ul>
               </div>`
            : '';

    const parties = data.parties && data.parties.length
        ? field('Parties', data.parties.map(p => `${p.name}<em> (${p.role})</em>`).join('<br>'))
        : '';

    return `<div class="lc-structured-data">
  <div class="lc-grid">
    ${field('Contract Type', data.contract_type)}
    ${parties}
    ${field('Effective Date', data.effective_date)}
    ${field('Term', data.term_duration)}
    ${field('Auto-Renewal', data.auto_renewal)}
    ${field('Payment Terms', data.payment_terms)}
    ${field('Termination Notice', data.termination_notice)}
    ${field('Governing Law', data.governing_law)}
    ${field('Liability Cap', data.liability_cap)}
  </div>
  ${list('Key Obligations', data.key_obligations)}
  ${list('Key Risks', data.key_risks, 'lc-risks')}
</div>`;
}

// Export to global scope

window.LegalInsightLC = {
    getRelevantChunks,
    calculateSemanticConsistency,
    formatAnalysisMessages,
    formatExtractionMessages,
    parseStructuredData,
    renderStructuredData,
    ready: true
};

window.dispatchEvent(new CustomEvent('lc:ready'));
console.log('[LegalInsight] LangChain.js engine ready');
