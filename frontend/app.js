/**
 * LegalInsight - Main Application
 *
 * Single path: every analysis, follow-up, and extraction goes through the
 * LegalInsight Self-RAG backend (retrieval + guardrails + self-healing
 * critique/retry). No provider dropdown, no client-side API key -- the
 * backend holds its own LLM API key server-side.
 */

// PDF.js setup

if (typeof pdfjsLib !== 'undefined') {
    pdfjsLib.GlobalWorkerOptions.workerSrc =
        'https://cdnjs.cloudflare.com/ajax/libs/pdf.js/3.11.174/pdf.worker.min.js';
}

// State

let currentPDFText = '';
let lastContractText = '';   // contract used for the current analysis (for follow-ups/extraction)
let analytics = { totalAnalyses: 0, totalTimeSaved: 0, totalEfficiency: 0 };

// The one backend this app talks to. Change this if you're running your own
// (e.g. 'http://localhost:5000' for local development against backend/api.py).
const BACKEND_URL = 'https://legalinsight-production.up.railway.app';

// Initialisation

window.addEventListener('DOMContentLoaded', function () {
    loadAnalytics();
    setupDragAndDrop();
    initializeBackend();
});

async function initializeBackend() {
    setStatusLine('Connecting to Self-RAG backend…', '');
    try {
        // The backend already initializes itself at boot (see backend/api.py's
        // __main__), so a healthy backend needs no further action here --
        // only fall back to POST /initialize if it somehow isn't ready yet.
        // Avoids forcing every single page load to redundantly reload the
        // embedding model.
        const health = await fetch(BACKEND_URL + '/health').then(r => r.json());
        if (health.model_loaded && health.retriever_loaded) {
            setStatusLine('', '');
            return;
        }

        const res = await fetch(BACKEND_URL + '/initialize', { method: 'POST' });
        const data = await res.json();

        const modelOk     = data.model && !data.model.error;
        const retrieverOk = data.retriever && !data.retriever.error;

        if (modelOk && retrieverOk) {
            setStatusLine('', '');  // ready -- no banner needed
        } else {
            const problems = [
                !modelOk     ? (data.model && data.model.message) || (data.model && data.model.error) : null,
                !retrieverOk ? (data.retriever && data.retriever.error) : null
            ].filter(Boolean).join(' | ');
            setStatusLine('⚠ Backend not fully ready: ' + (problems || 'unknown issue'), 'error');
        }
    } catch (err) {
        console.error('Backend init failed:', err);
        setStatusLine('✗ Could not reach the LegalInsight backend. Please try again shortly.', 'error');
    }
}

function setStatusLine(text, cls) {
    const el = document.getElementById('backend-status-line');
    if (!el) return;
    el.textContent = text;
    el.className = 'status-line' + (cls ? ' ' + cls : '');
    el.style.display = text ? 'block' : 'none';
}

// Drag-and-drop

function setupDragAndDrop() {
    const uploadBox = document.getElementById('upload-box');
    if (!uploadBox) return;

    uploadBox.addEventListener('dragover', e => {
        e.preventDefault();
        uploadBox.classList.add('dragover');
    });
    uploadBox.addEventListener('dragleave', () => uploadBox.classList.remove('dragover'));
    uploadBox.addEventListener('drop', e => {
        e.preventDefault();
        uploadBox.classList.remove('dragover');
        const file = e.dataTransfer.files[0];
        if (file && file.type === 'application/pdf') {
            handlePDFFile(file);
        } else {
            alert('Please drop a PDF file');
        }
    });
}

// PDF handling

async function handlePDFUpload(event) {
    const file = event.target.files[0];
    if (file && file.type === 'application/pdf') await handlePDFFile(file);
}

async function handlePDFFile(file) {
    if (typeof pdfjsLib === 'undefined') {
        alert('PDF.js library not loaded. Please refresh the page.');
        return;
    }
    try {
        setLoading(true, 'Extracting text from PDF...');
        const arrayBuffer = await file.arrayBuffer();
        const pdf = await pdfjsLib.getDocument({ data: arrayBuffer }).promise;

        let fullText = '';
        for (let p = 1; p <= pdf.numPages; p++) {
            const page    = await pdf.getPage(p);
            const content = await page.getTextContent();
            fullText += content.items.map(i => i.str).join(' ') + '\n\n';
        }

        currentPDFText = fullText;
        setValue('contract-text', fullText);
        setText('file-name', file.name);
        setText('file-pages', pdf.numPages);
        show('file-info');
        hide('upload-box');
    } catch (err) {
        console.error('PDF error:', err);
        alert('Error processing PDF. Please try again.');
    } finally {
        setLoading(false);
    }
}

function clearFile() {
    currentPDFText = '';
    const el = document.getElementById('pdf-upload');
    if (el) el.value = '';
    show('upload-box');
    hide('file-info');
}

// Main analysis flow

async function analyzeContract() {
    const contractText = getContractText();
    const query = getValue('query-text') ||
        'Analyse this legal contract. Provide a comprehensive summary including: ' +
        'key parties, main obligations, payment terms, important dates, ' +
        'termination clauses, and any notable risks or concerns.';

    if (!contractText) { alert('Please upload a PDF or paste contract text'); return; }
    await performAnalysis(contractText, query);
}

async function summarizeContract() {
    const contractText = getContractText();
    if (!contractText) { alert('Please upload a PDF or paste contract text'); return; }

    const query = 'Provide a concise summary of this legal contract, highlighting: ' +
        '1) Key parties, 2) Main obligations, 3) Important dates, ' +
        '4) Payment terms, 5) Termination conditions, 6) Notable clauses or risks.';
    await performAnalysis(contractText, query);
}

async function performAnalysis(contractText, query) {
    setLoading(true, 'Running self-healing retrieve → generate → critique loop...');
    hide('results-section');
    hide('followup-section');
    hide('structured-data-section');
    disableBtn('analyze-btn', true);

    lastContractText = contractText;

    try {
        const res = await fetch(BACKEND_URL + '/analyze_contract_self_healing', {
            method: 'POST',
            headers: { 'Content-Type': 'application/json' },
            body: JSON.stringify({ contract_text: contractText, query, max_attempts: 2 })
        });
        const data = await res.json();

        if (!res.ok) {
            const reasons = (data.reasons || []).join(', ');
            alert(
                (data.error || 'Analysis failed') +
                (reasons ? '\n\nReason(s): ' + reasons : '')
            );
            return;
        }

        displayResults(data, contractText);
        updateAnalytics(contractText.length, data.performance.total_time_seconds);

    } catch (err) {
        console.error('Analysis failed:', err);
        alert(
            'Could not reach the LegalInsight backend.\n\n' +
            'It may be waking up from idle -- please try again in a moment.\n\n' +
            'Details: ' + err.message
        );
    } finally {
        setLoading(false);
        disableBtn('analyze-btn', false);
    }
}

function displayResults(data, contractText) {
    const manualTime = estimateManualTime(contractText.length);
    const totalTime  = data.performance.total_time_seconds;
    const timeSaved  = manualTime - totalTime;
    const efficiency = timeSaved / manualTime * 100;
    const speedup    = manualTime / totalTime;

    setText('time-saved', (timeSaved / 60).toFixed(1) + ' min');
    setText('efficiency', efficiency.toFixed(1) + '%');
    setText('speedup',    speedup.toFixed(0) + 'x');

    const riskLabel = data.used_fallback ? 'Insufficient Info' : 'Grounded';
    setText('hallucination-risk', riskLabel);
    const hallCard = document.getElementById('hallucination-card');
    if (hallCard) {
        hallCard.className = 'metric-card ' + (data.used_fallback ? 'warning' : 'success');
    }

    setText('answer-content', data.answer);

    setText('manual-time',     (manualTime / 60).toFixed(1) + ' minutes');
    setText('ai-time',         totalTime.toFixed(2) + ' seconds');
    setText('contract-length', contractText.length.toLocaleString() + ' characters');
    setText('estimated-pages', Math.round(contractText.length / 3000) + ' pages');

    const guardEl = document.getElementById('guardrails-status');
    if (guardEl) {
        if (data.guardrails && data.guardrails.output_blocked) {
            guardEl.innerHTML =
                '<span class="error">⚠ Output withheld by guardrails: ' +
                escapeHtml(data.guardrails.output_blocked_reasons.join(', ')) + '</span>';
        } else {
            guardEl.innerHTML = '<span class="success">✓ Passed input and output checks</span>';
        }
    }

    const traceEl = document.getElementById('healing-trace-content');
    if (traceEl) {
        traceEl.innerHTML = data.trace.map((step, i) => {
            const label = step.accepted ? '✓ Accepted' : '↻ Rejected, retrying';
            const reasons = step.reasons && step.reasons.length
                ? '<br><em>' + escapeHtml(step.reasons.join('; ')) + '</em>' : '';
            return `<div class="alternative-item">` +
                `<strong>Attempt ${i + 1} (${label}):</strong> ` +
                `<span style="opacity:0.7">${escapeHtml(step.retrieval_query)}</span>` +
                reasons +
                `</div>`;
        }).join('');
    }

    show('results-section');
    document.getElementById('results-section').scrollIntoView({ behavior: 'smooth' });
    show('followup-section');
    hide('structured-data-section');
}

function toggleHealingTrace() {
    const el = document.getElementById('healing-trace-content');
    if (el) el.style.display = el.style.display === 'none' ? 'block' : 'none';
}

// Follow-up questions
//
// Each follow-up is a fresh self-healing analysis of the same contract with
// the new question -- retrieval finds the relevant clause per-question, so
// no client-side conversation history needs to be threaded through.

async function askFollowUp() {
    const input = getValue('followup-input').trim();
    if (!input) { alert('Please enter a follow-up question'); return; }
    if (!lastContractText) { alert('Please run an analysis first'); return; }

    show('followup-loading');
    disableBtn('followup-btn', true);

    try {
        const res = await fetch(BACKEND_URL + '/analyze_contract_self_healing', {
            method: 'POST',
            headers: { 'Content-Type': 'application/json' },
            body: JSON.stringify({ contract_text: lastContractText, query: input, max_attempts: 2 })
        });
        const data = await res.json();

        if (!res.ok) {
            const reasons = (data.reasons || []).join(', ');
            alert((data.error || 'Follow-up failed') + (reasons ? '\n\nReason(s): ' + reasons : ''));
            return;
        }

        appendFollowUpExchange(input, data.answer);
        setValue('followup-input', '');

    } catch (err) {
        console.error('Follow-up failed:', err);
        alert('Follow-up failed: ' + err.message);
    } finally {
        hide('followup-loading');
        disableBtn('followup-btn', false);
    }
}

function appendFollowUpExchange(question, answer) {
    const container = document.getElementById('conversation-history');
    if (!container) return;

    const div = document.createElement('div');
    div.className = 'conv-exchange';
    div.innerHTML =
        `<div class="conv-q"><span class="conv-role">You</span>${escapeHtml(question)}</div>` +
        `<div class="conv-a"><span class="conv-role">Analysis</span>${escapeHtml(answer)}</div>`;
    container.appendChild(div);
    div.scrollIntoView({ behavior: 'smooth' });
}

function clearConversation() {
    const container = document.getElementById('conversation-history');
    if (container) container.innerHTML = '';
}

// Structured key-terms extraction

async function extractKeyTerms() {
    const contractText = lastContractText || getContractText();
    if (!contractText) { alert('Please run an analysis first'); return; }

    disableBtn('extract-btn', true);
    const btn = document.getElementById('extract-btn');
    if (btn) btn.textContent = 'Extracting…';

    try {
        const res = await fetch(BACKEND_URL + '/extract_key_terms', {
            method: 'POST',
            headers: { 'Content-Type': 'application/json' },
            body: JSON.stringify({ contract_text: contractText })
        });
        const result = await res.json();

        if (!res.ok) {
            alert((result.error || 'Key terms extraction failed'));
            return;
        }

        const content = document.getElementById('structured-data-content');
        if (content) content.innerHTML = renderStructuredData(result.data);

        show('structured-data-section');
        document.getElementById('structured-data-section').scrollIntoView({ behavior: 'smooth' });

    } catch (err) {
        console.error('Extraction failed:', err);
        alert('Key terms extraction failed: ' + err.message);
    } finally {
        disableBtn('extract-btn', false);
        const btn2 = document.getElementById('extract-btn');
        if (btn2) btn2.textContent = '🗂️ Extract Key Terms';
    }
}

function renderStructuredData(data) {
    if (!data) return '<p>No structured data available.</p>';
    const list = (label, items) => {
        if (!items) return '';
        const arr = Array.isArray(items) ? items : [items];
        if (!arr.length) return '';
        return `<p><strong>${label}:</strong> ${arr.map(escapeHtml).join(', ')}</p>`;
    };
    return [
        list('Parties', data.parties),
        list('Dates', data.dates),
        data.payment_terms ? `<p><strong>Payment Terms:</strong> ${escapeHtml(String(data.payment_terms))}</p>` : '',
        data.liability_cap ? `<p><strong>Liability Cap:</strong> ${escapeHtml(String(data.liability_cap))}</p>` : '',
        list('Risks', data.risks),
    ].filter(Boolean).join('') || '<p>Could not parse structured data.</p>';
}

// Analytics

function estimateManualTime(contractLength) {
    return (contractLength / 3000) * 7.5 * 60;  // seconds
}

function updateAnalytics(contractLength, analysisTime) {
    const manual    = estimateManualTime(contractLength);
    const saved     = manual - analysisTime;
    const eff       = (saved / manual) * 100;

    analytics.totalAnalyses++;
    analytics.totalTimeSaved += saved / 60;
    analytics.totalEfficiency =
        ((analytics.totalEfficiency * (analytics.totalAnalyses - 1)) + eff) / analytics.totalAnalyses;

    saveAnalytics();
    displayAnalytics();
}

function displayAnalytics() {
    setText('total-analyses',  analytics.totalAnalyses);
    setText('total-time-saved', analytics.totalTimeSaved.toFixed(1));
    setText('avg-efficiency',  analytics.totalEfficiency.toFixed(1) + '%');
    if (analytics.totalAnalyses > 0) show('analytics-section');
}

function saveAnalytics()  { localStorage.setItem('analytics', JSON.stringify(analytics)); }
function loadAnalytics()  {
    const saved = localStorage.getItem('analytics');
    if (saved) { analytics = JSON.parse(saved); displayAnalytics(); }
}
function clearAnalytics() {
    if (!confirm('Clear all analytics?')) return;
    analytics = { totalAnalyses: 0, totalTimeSaved: 0, totalEfficiency: 0 };
    saveAnalytics();
    displayAnalytics();
}
function refreshAnalytics() { displayAnalytics(); }

// Utility functions

function getContractText() {
    const el = document.getElementById('contract-text');
    return el ? el.value.trim() : '';
}
function getValue(id)       { const el = document.getElementById(id); return el ? el.value : ''; }
function setValue(id, val)  { const el = document.getElementById(id); if (el) el.value = val; }
function getText(id)        { const el = document.getElementById(id); return el ? el.textContent : ''; }
function setText(id, val)   { const el = document.getElementById(id); if (el) el.textContent = val; }
function show(id)           { const el = document.getElementById(id); if (el) el.style.display = 'block'; }
function hide(id)           { const el = document.getElementById(id); if (el) el.style.display = 'none'; }
function disableBtn(id, v)  { const el = document.getElementById(id); if (el) el.disabled = v; }
function setLoading(on, text) {
    const l  = document.getElementById('loading');
    const lt = document.getElementById('loading-text');
    if (l)  l.style.display  = on ? 'block' : 'none';
    if (lt && text) lt.textContent = text;
}

function escapeHtml(str) {
    return String(str)
        .replace(/&/g,  '&amp;')
        .replace(/</g,  '&lt;')
        .replace(/>/g,  '&gt;')
        .replace(/"/g,  '&quot;')
        .replace(/'/g,  '&#39;');
}

function clearAll() {
    setValue('contract-text', '');
    setValue('query-text', '');
    clearFile();
    clearConversation();
    hide('results-section');
    hide('followup-section');
    hide('structured-data-section');
    lastContractText = '';
}

function loadExample() {
    const exampleContract =
`SERVICE AGREEMENT

This Service Agreement ("Agreement") is entered into as of January 15, 2024 ("Effective Date"), by and between:

TechCorp Solutions Inc., a Delaware corporation with offices at 123 Innovation Drive, San Francisco, CA 94105 ("Provider")

AND

Global Enterprises LLC, a California limited liability company with offices at 456 Business Boulevard, Los Angeles, CA 90001 ("Client")

1. SERVICES
Provider agrees to provide software development and consulting services as detailed in Exhibit A ("Services"). Services include custom application development, system integration, and technical support.

2. TERM
This Agreement shall commence on the Effective Date and continue for a period of twelve (12) months, unless earlier terminated as provided herein ("Initial Term"). The Agreement shall automatically renew for successive one (1) year periods unless either party provides written notice of non-renewal at least sixty (60) days prior to the end of the then-current term.

3. COMPENSATION
Client shall pay Provider a monthly fee of $25,000 USD, payable within fifteen (15) days of invoice receipt. Late payments shall accrue interest at 1.5% per month.

4. CONFIDENTIALITY
Both parties agree to maintain confidentiality of all proprietary information disclosed during the term of this Agreement and for three (3) years thereafter.

5. TERMINATION
Either party may terminate this Agreement with thirty (30) days written notice. Provider may terminate immediately if Client fails to pay any invoice within forty-five (45) days of receipt.

6. LIABILITY
Provider's total liability under this Agreement shall not exceed the total fees paid by Client in the twelve (12) months preceding the claim.

7. GOVERNING LAW
This Agreement shall be governed by the laws of the State of California.

IN WITNESS WHEREOF, the parties have executed this Agreement as of the Effective Date.`;

    setValue('contract-text', exampleContract);
    clearFile();
}
