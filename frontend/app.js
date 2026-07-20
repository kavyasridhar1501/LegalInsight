/**
 * LegalInsight - Main Application
 *
 * Works with langchain-engine.js (loaded as ES module) when available.
 * Falls back gracefully if the LangChain module hasn't loaded yet.
 */

// PDF.js setup

if (typeof pdfjsLib !== 'undefined') {
    pdfjsLib.GlobalWorkerOptions.workerSrc =
        'https://cdnjs.cloudflare.com/ajax/libs/pdf.js/3.11.174/pdf.worker.min.js';
}

// State

let currentPDFText = '';
let conversationHistory = [];   // [{role: 'user'|'assistant', content: string}]
let lastContractText   = '';    // contract used for current analysis (for follow-ups)
let analytics = { totalAnalyses: 0, totalTimeSaved: 0, totalEfficiency: 0 };

// Fill in once the backend is deployed (e.g. to Railway) so visitors get a
// working Self-RAG backend out of the box, with no setup of their own.
// Leave empty to require everyone to enter their own backend URL.
const DEFAULT_BACKEND_URL = '';

// LangChain engine accessor

/**
 * Returns the LangChain engine once the module has loaded.
 * Falls back to null after 6 s so the app still works without it.
 */
async function getLCEngine(timeout = 6000) {
    if (window.LegalInsightLC) return window.LegalInsightLC;
    return new Promise(resolve => {
        const timer = setTimeout(() => resolve(null), timeout);
        window.addEventListener('lc:ready', () => {
            clearTimeout(timer);
            resolve(window.LegalInsightLC);
        }, { once: true });
    });
}

// Initialisation

window.addEventListener('DOMContentLoaded', function () {
    loadSettings();
    loadAnalytics();
    setupDragAndDrop();

    // If a deployed backend is configured and it's the active provider,
    // initialise it automatically so first-time visitors don't have to
    // find and click the button before anything works.
    if (DEFAULT_BACKEND_URL && getValue('api-provider') === 'reliability-backend') {
        initializeBackend();
    }
});

function loadSettings() {
    const provider = localStorage.getItem('api_provider') || 'reliability-backend';
    const apiKey    = localStorage.getItem('api_key') || '';

    // Seed localStorage with the deployed default on first visit, so the
    // Self-RAG backend works immediately with no setup -- not just prefill
    // the input box (which alone wouldn't persist until "Save URL" is clicked).
    if (!localStorage.getItem('backend_url') && DEFAULT_BACKEND_URL) {
        localStorage.setItem('backend_url', DEFAULT_BACKEND_URL);
    }
    const backendUrl = localStorage.getItem('backend_url') || '';

    const providerEl = document.getElementById('api-provider');
    if (providerEl) providerEl.value = provider;

    const apiKeyEl = document.getElementById('api-key');
    if (apiKey && apiKeyEl) {
        apiKeyEl.value = apiKey;
        const statusEl = document.getElementById('api-status');
        if (statusEl) { statusEl.textContent = '✓ Saved'; statusEl.className = 'success'; }
    }

    const backendUrlEl = document.getElementById('backend-url');
    if (backendUrlEl) backendUrlEl.value = backendUrl;

    updateApiConfig();
}

function updateApiConfig() {
    const providerEl = document.getElementById('api-provider');
    if (!providerEl) return;

    const provider = providerEl.value;
    localStorage.setItem('api_provider', provider);

    const apiKeySection    = document.getElementById('api-key-section');
    const backendUrlSection = document.getElementById('backend-url-section');
    const apiLink          = document.getElementById('api-link');

    if (provider === 'reliability-backend') {
        if (apiKeySection) apiKeySection.style.display = 'none';
        if (backendUrlSection) backendUrlSection.style.display = 'block';
        return;
    }
    if (backendUrlSection) backendUrlSection.style.display = 'none';

    if (provider === 'demo') {
        if (apiKeySection) apiKeySection.style.display = 'none';
    } else {
        if (apiKeySection) apiKeySection.style.display = 'block';
        const links = {
            openai:    { url: 'https://platform.openai.com/api-keys',          name: 'OpenAI' },
            anthropic: { url: 'https://console.anthropic.com/account/keys',    name: 'Anthropic' },
            gemini:    { url: 'https://makersuite.google.com/app/apikey',       name: 'Google AI Studio' },
            groq:      { url: 'https://console.groq.com/keys',                 name: 'Groq' },
            cohere:    { url: 'https://dashboard.cohere.com/api-keys',         name: 'Cohere' },
            mistral:   { url: 'https://console.mistral.ai/api-keys',           name: 'Mistral AI' }
        };
        if (apiLink && links[provider]) {
            apiLink.href = links[provider].url;
            apiLink.textContent = links[provider].name;
        }
    }
}

function saveApiKey() {
    const apiKeyEl = document.getElementById('api-key');
    const statusEl = document.getElementById('api-status');
    if (!apiKeyEl || !statusEl) return;

    const apiKey = apiKeyEl.value.trim();
    if (!apiKey) {
        statusEl.textContent = '✗ Please enter an API key';
        statusEl.className = 'error';
        return;
    }
    localStorage.setItem('api_key', apiKey);
    statusEl.textContent = '✓ API Key Saved';
    statusEl.className = 'success';
}

function saveBackendUrl() {
    const urlEl    = document.getElementById('backend-url');
    const statusEl = document.getElementById('backend-status');
    if (!urlEl || !statusEl) return;

    const url = urlEl.value.trim().replace(/\/+$/, '');
    if (!url) {
        statusEl.textContent = '✗ Please enter a backend URL';
        statusEl.className = 'error';
        return;
    }
    localStorage.setItem('backend_url', url);
    statusEl.textContent = '✓ URL Saved — click "Initialize Backend" next';
    statusEl.className = 'success';
}

async function initializeBackend() {
    const statusEl = document.getElementById('backend-status');
    const backendUrl = localStorage.getItem('backend_url');
    if (!backendUrl) {
        alert('Please enter and save a backend URL first');
        return;
    }

    disableBtn('init-backend-btn', true);
    if (statusEl) { statusEl.textContent = 'Initialising model + retriever…'; statusEl.className = ''; }

    try {
        const res = await fetch(backendUrl + '/initialize', { method: 'POST' });
        const data = await res.json();

        const modelOk     = data.model && !data.model.error;
        const retrieverOk = data.retriever && !data.retriever.error;

        if (modelOk && retrieverOk) {
            if (statusEl) { statusEl.textContent = '✓ Backend ready'; statusEl.className = 'success'; }
        } else {
            const problems = [
                !modelOk     ? (data.model && data.model.message) || (data.model && data.model.error) : null,
                !retrieverOk ? (data.retriever && data.retriever.error) : null
            ].filter(Boolean).join(' | ');
            if (statusEl) { statusEl.textContent = '⚠ ' + (problems || 'Backend not fully initialised'); statusEl.className = 'error'; }
        }
    } catch (err) {
        console.error('Backend init failed:', err);
        if (statusEl) {
            statusEl.textContent = '✗ Could not reach backend at ' + backendUrl;
            statusEl.className = 'error';
        }
    } finally {
        disableBtn('init-backend-btn', false);
    }
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
    const provider = getValue('api-provider') || 'demo';

    if (provider === 'reliability-backend') {
        await performBackendAnalysis(contractText, query);
        return;
    }

    const apiKey = localStorage.getItem('api_key');
    if (provider !== 'demo' && !apiKey) {
        alert('Please enter your API key first');
        return;
    }

    setLoading(true, 'Initialising analysis engine...');
    hide('results-section');
    hide('followup-section');
    hide('structured-data-section');
    disableBtn('analyze-btn', true);

    // Reset conversation for fresh analysis
    conversationHistory = [];
    lastContractText    = contractText;

    const startTime = Date.now();

    try {
        const lc = await getLCEngine();

        // Step 1: chunk + find relevant context
        let context = contractText;
        let chunkInfo = null;

        if (lc) {
            setLoading(true, 'Chunking contract with LangChain...');
            const result = await lc.getRelevantChunks(contractText, query);
            context   = result.context;
            chunkInfo = { used: result.used, total: result.total };
        }

        // Step 2: format messages with LangChain prompt template
        let messagesTemplate = null;
        if (lc) {
            setLoading(true, 'Formatting prompt with LangChain...');
            messagesTemplate = await lc.formatAnalysisMessages(context, query);
        }

        // Step 3: three generations for consistency scoring
        setLoading(true, 'Generating analysis (1/3)...');
        const r1 = await callAI(provider, apiKey, contractText, query, 0.1, messagesTemplate);

        setLoading(true, 'Generating verification (2/3)...');
        const r2 = await callAI(provider, apiKey, contractText, query, 0.5, messagesTemplate);

        setLoading(true, 'Generating verification (3/3)...');
        const r3 = await callAI(provider, apiKey, contractText, query, 0.9, messagesTemplate);

        const totalTime = (Date.now() - startTime) / 1000;

        // Step 4: consistency score
        const consistencyScore = lc
            ? lc.calculateSemanticConsistency([r1, r2, r3])
            : calculateConsistencyFallback([r1, r2, r3]);

        // Step 5: store primary in conversation history
        conversationHistory.push({ role: 'user',      content: query });
        conversationHistory.push({ role: 'assistant', content: r1   });

        displayResults({
            answer: r1,
            alternativeResponses: [r2, r3],
            consistencyScore,
            contractLength: contractText.length,
            analysisTime: totalTime,
            chunkInfo,
            lcEnabled: !!lc
        });

        updateAnalytics(contractText.length, totalTime);

    } catch (err) {
        console.error('Analysis failed:', err);
        alert('Analysis failed: ' + err.message + '\n\nPlease check your API key and try again.');
    } finally {
        setLoading(false);
        disableBtn('analyze-btn', false);
    }
}

// Reliability backend (self-healing RAG + guardrails)

async function performBackendAnalysis(contractText, query) {
    const backendUrl = localStorage.getItem('backend_url');
    if (!backendUrl) {
        alert('Please enter and save a Reliability Backend URL first');
        return;
    }

    setLoading(true, 'Running self-healing retrieve → generate → critique loop...');
    hide('results-section');
    hide('followup-section');
    hide('structured-data-section');
    disableBtn('analyze-btn', true);

    conversationHistory = [];
    lastContractText    = contractText;

    try {
        const res = await fetch(backendUrl + '/analyze_contract_self_healing', {
            method: 'POST',
            headers: { 'Content-Type': 'application/json' },
            body: JSON.stringify({ contract_text: contractText, query, max_attempts: 2 })
        });
        const data = await res.json();

        if (!res.ok) {
            const reasons = (data.reasons || []).join(', ');
            alert(
                (data.error || 'Backend request failed') +
                (reasons ? '\n\nReason(s): ' + reasons : '')
            );
            return;
        }

        displayBackendResults(data, contractText);
        updateAnalytics(contractText.length, data.performance.total_time_seconds);

    } catch (err) {
        console.error('Backend analysis failed:', err);
        alert(
            'Could not reach the reliability backend at ' + backendUrl + '.\n\n' +
            'Make sure backend/api.py is running and reachable, and that you clicked ' +
            '"Initialize Backend" first.\n\nDetails: ' + err.message
        );
    } finally {
        setLoading(false);
        disableBtn('analyze-btn', false);
    }
}

function displayBackendResults(data, contractText) {
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

    hide('chunk-info');
    hide('eigenscore-section');
    hide('alternatives-section');
    show('reliability-section');

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
    hide('followup-section');
    hide('structured-data-section');
}

function toggleHealingTrace() {
    const el = document.getElementById('healing-trace-content');
    if (el) el.style.display = el.style.display === 'none' ? 'block' : 'none';
}

// Follow-up questions

async function askFollowUp() {
    const input = getValue('followup-input').trim();
    if (!input) { alert('Please enter a follow-up question'); return; }

    const provider = getValue('api-provider') || 'demo';
    const apiKey   = localStorage.getItem('api_key');
    if (provider !== 'demo' && !apiKey) {
        alert('Please enter your API key first');
        return;
    }

    show('followup-loading');
    disableBtn('followup-btn', true);

    try {
        const lc = await getLCEngine();

        let messages = null;
        if (lc && lastContractText) {
            const result = await lc.getRelevantChunks(lastContractText, input, 3);
            messages = await lc.formatAnalysisMessages(
                result.context,
                input,
                conversationHistory.slice(-6)  // last 3 exchanges
            );
        }

        const response = await callAI(provider, apiKey, lastContractText, input, 0.2, messages);

        conversationHistory.push({ role: 'user',      content: input   });
        conversationHistory.push({ role: 'assistant', content: response });

        appendFollowUpExchange(input, response);
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
    conversationHistory = [];
    const container = document.getElementById('conversation-history');
    if (container) container.innerHTML = '';
}

// Structured key-terms extraction

async function extractKeyTerms() {
    const contractText = lastContractText || getContractText();
    if (!contractText) { alert('Please run an analysis first'); return; }

    const provider = getValue('api-provider') || 'demo';
    const apiKey   = localStorage.getItem('api_key');
    if (provider !== 'demo' && !apiKey) {
        alert('Please enter your API key first');
        return;
    }

    disableBtn('extract-btn', true);
    const btn = document.getElementById('extract-btn');
    if (btn) btn.textContent = 'Extracting…';

    try {
        const lc = await getLCEngine();
        let messages = null;

        if (lc) {
            messages = await lc.formatExtractionMessages(contractText);
        } else {
            // Fallback: simple extraction prompt without LangChain
            messages = [
                { role: 'system', content: 'You are a legal data-extraction specialist. Return ONLY valid JSON.' },
                { role: 'user',   content: buildFallbackExtractionPrompt(contractText) }
            ];
        }

        const rawJson = await callAI(provider, apiKey, contractText, '', 0.1, messages);

        const data = lc
            ? lc.parseStructuredData(rawJson)
            : JSON.parse(rawJson.replace(/```json\n?|```\n?/g, '').trim());

        const html = lc
            ? lc.renderStructuredData(data)
            : renderFallbackStructuredData(data);

        const content = document.getElementById('structured-data-content');
        if (content) content.innerHTML = html || '<p>Could not parse structured data.</p>';

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

function buildFallbackExtractionPrompt(text) {
    return `Extract contract data as JSON with fields: contract_type, parties, effective_date, ` +
           `term_duration, payment_terms, termination_notice, governing_law, ` +
           `key_obligations (array), key_risks (array).\n\nContract:\n${text.slice(0, 4000)}`;
}

function renderFallbackStructuredData(data) {
    if (!data) return '<p>No structured data available.</p>';
    return `<pre style="white-space:pre-wrap;font-size:0.9em">${JSON.stringify(data, null, 2)}</pre>`;
}

// Provider routing

/**
 * Routes to the appropriate provider.
 *
 * @param {string}      provider
 * @param {string}      apiKey
 * @param {string}      contractText  - raw text (used only for demo / Gemini concat)
 * @param {string}      query         - raw query (used only for demo)
 * @param {number}      temperature
 * @param {Array|null}  messages      - pre-formatted LangChain messages (preferred)
 */
async function callAI(provider, apiKey, contractText, query, temperature, messages) {
    if (provider === 'demo') {
        return generateDemoResponse(contractText, query);
    }

    // If LangChain didn't produce messages (engine not loaded), build them inline
    const msgs = messages || buildFallbackMessages(contractText, query);

    switch (provider) {
        case 'openai':    return callOpenAI(apiKey, msgs, temperature);
        case 'anthropic': return callAnthropic(apiKey, msgs, temperature);
        case 'gemini':    return callGemini(apiKey, msgs, temperature);
        case 'groq':      return callGroq(apiKey, msgs, temperature);
        case 'cohere':    return callCohere(apiKey, msgs, temperature);
        case 'mistral':   return callMistral(apiKey, msgs, temperature);
        default:          throw new Error('Unknown provider: ' + provider);
    }
}

/** Builds a minimal system+user message pair when LangChain is unavailable. */
function buildFallbackMessages(contractText, query) {
    return [
        {
            role: 'system',
            content: 'You are a legal expert specialising in contract analysis. ' +
                     'Provide detailed, accurate analysis.'
        },
        {
            role: 'user',
            content: `Contract:\n\n${contractText}\n\nQuestion: ${query}`
        }
    ];
}

// Provider implementations

async function callOpenAI(apiKey, messages, temperature) {
    const res = await fetch('https://api.openai.com/v1/chat/completions', {
        method: 'POST',
        headers: {
            'Content-Type': 'application/json',
            'Authorization': 'Bearer ' + apiKey
        },
        body: JSON.stringify({
            model: 'gpt-4o',
            messages,
            temperature,
            max_tokens: 1500
        })
    });
    if (!res.ok) {
        const err = await res.json();
        throw new Error((err.error && err.error.message) || 'OpenAI API error');
    }
    const data = await res.json();
    return data.choices[0].message.content;
}

async function callAnthropic(apiKey, messages, temperature) {
    // Anthropic requires system as a top-level field, not in messages[]
    const systemMsg  = messages.find(m => m.role === 'system');
    const chatMsgs   = messages.filter(m => m.role !== 'system');

    const body = {
        model: 'claude-3-5-sonnet-20241022',
        max_tokens: 1500,
        temperature,
        messages: chatMsgs
    };
    if (systemMsg) body.system = systemMsg.content;

    const res = await fetch('https://api.anthropic.com/v1/messages', {
        method: 'POST',
        headers: {
            'Content-Type': 'application/json',
            'x-api-key': apiKey,
            'anthropic-version': '2023-06-01',
            'anthropic-dangerous-allow-browser': 'true'
        },
        body: JSON.stringify(body)
    });
    if (!res.ok) {
        const err = await res.json();
        throw new Error((err.error && err.error.message) || 'Anthropic API error');
    }
    const data = await res.json();
    return data.content[0].text;
}

async function callGemini(apiKey, messages, temperature) {
    // Gemini uses a different format; combine system + user into a single prompt
    const systemMsg = messages.find(m => m.role === 'system');
    const userMsgs  = messages.filter(m => m.role === 'user');
    const lastUser  = userMsgs[userMsgs.length - 1];

    const prefix = systemMsg ? systemMsg.content + '\n\n' : '';

    // Build multi-turn history for Gemini if conversation exists
    const history = messages
        .filter(m => m.role === 'user' || m.role === 'assistant')
        .slice(0, -1)
        .map(m => ({
            role: m.role === 'user' ? 'user' : 'model',
            parts: [{ text: m.content }]
        }));

    const contents = [
        ...history,
        { role: 'user', parts: [{ text: prefix + lastUser.content }] }
    ];

    const res = await fetch(
        `https://generativelanguage.googleapis.com/v1beta/models/gemini-1.5-pro-latest:generateContent?key=${apiKey}`,
        {
            method: 'POST',
            headers: { 'Content-Type': 'application/json' },
            body: JSON.stringify({
                contents,
                generationConfig: { temperature, maxOutputTokens: 1500 }
            })
        }
    );
    if (!res.ok) {
        const err = await res.json();
        throw new Error((err.error && err.error.message) || 'Gemini API error');
    }
    const data = await res.json();
    return data.candidates[0].content.parts[0].text;
}

async function callGroq(apiKey, messages, temperature) {
    const res = await fetch('https://api.groq.com/openai/v1/chat/completions', {
        method: 'POST',
        headers: {
            'Content-Type': 'application/json',
            'Authorization': 'Bearer ' + apiKey
        },
        body: JSON.stringify({
            model: 'llama-3.1-8b-instant',
            messages,
            temperature,
            max_tokens: 1500
        })
    });
    if (!res.ok) {
        const err = await res.json();
        throw new Error((err.error && err.error.message) || 'Groq API error');
    }
    const data = await res.json();
    return data.choices[0].message.content;
}

async function callCohere(apiKey, messages, temperature) {
    const systemMsg = messages.find(m => m.role === 'system');
    const userMsgs  = messages.filter(m => m.role !== 'system');
    const lastUser  = userMsgs[userMsgs.length - 1];

    // Convert history to Cohere format (USER/CHATBOT, uppercase)
    const chatHistory = userMsgs.slice(0, -1).map(m => ({
        role: m.role === 'user' ? 'USER' : 'CHATBOT',
        message: m.content
    }));

    const body = {
        model: 'command-r-plus',
        message: lastUser.content,
        temperature,
        max_tokens: 1500
    };
    if (systemMsg)              body.preamble     = systemMsg.content;
    if (chatHistory.length > 0) body.chat_history = chatHistory;

    const res = await fetch('https://api.cohere.ai/v1/chat', {
        method: 'POST',
        headers: {
            'Content-Type': 'application/json',
            'Authorization': 'Bearer ' + apiKey
        },
        body: JSON.stringify(body)
    });
    if (!res.ok) {
        const err = await res.json();
        throw new Error(err.message || 'Cohere API error');
    }
    const data = await res.json();
    return data.text;
}

async function callMistral(apiKey, messages, temperature) {
    const res = await fetch('https://api.mistral.ai/v1/chat/completions', {
        method: 'POST',
        headers: {
            'Content-Type': 'application/json',
            'Authorization': 'Bearer ' + apiKey
        },
        body: JSON.stringify({
            model: 'mistral-large-latest',
            messages,
            temperature,
            max_tokens: 1500
        })
    });
    if (!res.ok) {
        const err = await res.json();
        throw new Error((err.error && err.error.message) || 'Mistral API error');
    }
    const data = await res.json();
    return data.choices[0].message.content;
}

// Demo mode

function generateDemoResponse(contractText, query) {
    return new Promise(resolve => {
        setTimeout(() => {
            const type       = detectContractType(contractText);
            const pages      = Math.round(contractText.length / 3000);
            const complexity = contractText.length > 10000 ? 'High'
                             : contractText.length > 5000  ? 'Medium' : 'Low';

            resolve(
                'DEMO MODE ANALYSIS\n\nThis is a demonstration of LegalInsight.\n\n' +
                'Key Findings:\n' +
                `• Contract Type: ${type}\n` +
                `• Length: ${contractText.length.toLocaleString()} characters (~${pages} pages)\n` +
                `• Complexity: ${complexity}\n\n` +
                'To get AI-powered analysis:\n' +
                '1. Select a provider from the dropdown\n' +
                '2. Enter your API key\n' +
                '3. Click "Analyse Contract"\n\n' +
                'Demo mode does not provide real legal analysis.'
            );
        }, 1200);
    });
}

function detectContractType(text) {
    const t = text.toLowerCase();
    if (t.includes('employment') || t.includes('employee')) return 'Employment Agreement';
    if (t.includes('lease')      || t.includes('rent'))      return 'Lease Agreement';
    if (t.includes('license')    || t.includes('software'))  return 'Software License';
    if (t.includes('service')    || t.includes('consulting'))return 'Service Agreement';
    if (t.includes('purchase')   || t.includes('sale'))      return 'Purchase Agreement';
    return 'General Contract';
}

// Consistency scoring (fallback)

function calculateConsistencyFallback(responses) {
    const lengths   = responses.map(r => r.length);
    const avg       = lengths.reduce((a, b) => a + b, 0) / lengths.length;
    const variance  = Math.max(...lengths) - Math.min(...lengths);
    return Math.min(100, Math.max(60, 100 - (variance / avg * 100)));
}

// Display logic

function displayResults(data) {
    hide('reliability-section');
    show('eigenscore-section');
    show('alternatives-section');

    const manualTime = estimateManualTime(data.contractLength);
    const timeSaved  = manualTime - data.analysisTime;
    const efficiency = timeSaved / manualTime * 100;
    const speedup    = manualTime / data.analysisTime;

    setText('time-saved',  (timeSaved / 60).toFixed(1) + ' min');
    setText('efficiency',  efficiency.toFixed(1) + '%');
    setText('speedup',     speedup.toFixed(0) + 'x');

    const riskLabel = data.consistencyScore > 85 ? 'Low'
                    : data.consistencyScore > 70 ? 'Medium' : 'High';
    setText('hallucination-risk', riskLabel);

    const hallCard = document.getElementById('hallucination-card');
    if (hallCard) {
        hallCard.className = 'metric-card ' +
            (riskLabel === 'Low' ? 'success' : riskLabel === 'Medium' ? 'warning' : 'danger');
    }

    setText('answer-content', data.answer);
    setText('consistency-score', data.consistencyScore.toFixed(0) + '%');

    const fill = document.getElementById('consistency-fill');
    if (fill) fill.style.width = data.consistencyScore + '%';

    const interp = data.consistencyScore > 85
        ? 'Excellent – All responses highly consistent'
        : data.consistencyScore > 70
        ? 'Good – Responses mostly consistent'
        : 'Fair – Some variation detected, verify carefully';
    setText('consistency-interpretation', interp);

    const badge = document.getElementById('consistency-method');
    if (badge) {
        badge.textContent = data.lcEnabled
            ? 'Scored via LangChain Jaccard similarity'
            : 'Scored via length variance (LangChain loading…)';
    }

    setText('manual-time',      (manualTime / 60).toFixed(1) + ' minutes');
    setText('ai-time',          data.analysisTime.toFixed(2) + ' seconds');
    setText('contract-length',  data.contractLength.toLocaleString() + ' characters');
    setText('estimated-pages',  Math.round(data.contractLength / 3000) + ' pages');

    const chunkEl = document.getElementById('chunk-info');
    if (chunkEl) {
        if (data.chunkInfo && data.chunkInfo.total > 1) {
            chunkEl.textContent =
                `Analysed ${data.chunkInfo.used} of ${data.chunkInfo.total} sections ` +
                `(LangChain RAG chunking)`;
            chunkEl.style.display = 'inline-block';
        } else {
            chunkEl.style.display = 'none';
        }
    }

    const altDiv = document.getElementById('alternatives-content');
    if (altDiv) {
        altDiv.innerHTML = '';
        data.alternativeResponses.forEach((r, i) => {
            const d = document.createElement('div');
            d.className = 'alternative-item';
            d.innerHTML = `<strong>Verification Response ${i + 1}:</strong><br>${escapeHtml(r)}`;
            altDiv.appendChild(d);
        });
    }

    show('results-section');
    document.getElementById('results-section').scrollIntoView({ behavior: 'smooth' });

    show('followup-section');
    hide('structured-data-section');
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

function toggleAlternatives() {
    const el = document.getElementById('alternatives-content');
    if (el) el.style.display = el.style.display === 'none' ? 'block' : 'none';
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
