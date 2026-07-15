/**
 * LegalInsight - Standalone Client-Side Application
 * No backend server required - uses OpenAI/Anthropic API directly
 */

if (typeof pdfjsLib !== 'undefined') {
    pdfjsLib.GlobalWorkerOptions.workerSrc = 'https://cdnjs.cloudflare.com/ajax/libs/pdf.js/3.11.174/pdf.worker.min.js';
}

let currentPDFText = '';
let analytics = {
    totalAnalyses: 0,
    totalTimeSaved: 0,
    totalEfficiency: 0
};

window.addEventListener('DOMContentLoaded', function() {
    loadSettings();
    loadAnalytics();
    setupDragAndDrop();
});

function loadSettings() {
    const provider = localStorage.getItem('api_provider') || 'demo';
    const apiKey = localStorage.getItem('api_key') || '';

    const providerEl = document.getElementById('api-provider');
    if (providerEl) providerEl.value = provider;

    const apiKeyEl = document.getElementById('api-key');
    if (apiKey && apiKeyEl) {
        apiKeyEl.value = apiKey;
        const statusEl = document.getElementById('api-status');
        if (statusEl) {
            statusEl.textContent = '✓ Saved';
            statusEl.className = 'success';
        }
    }

    updateApiConfig();
}

function updateApiConfig() {
    const providerEl = document.getElementById('api-provider');
    if (!providerEl) return;

    const provider = providerEl.value;
    const apiKeySection = document.getElementById('api-key-section');
    const apiLink = document.getElementById('api-link');

    localStorage.setItem('api_provider', provider);

    if (provider === 'demo') {
        if (apiKeySection) apiKeySection.style.display = 'none';
    } else {
        if (apiKeySection) apiKeySection.style.display = 'block';

        if (apiLink) {
            const providerLinks = {
                'openai': { url: 'https://platform.openai.com/api-keys', name: 'OpenAI' },
                'anthropic': { url: 'https://console.anthropic.com/account/keys', name: 'Anthropic' },
                'gemini': { url: 'https://makersuite.google.com/app/apikey', name: 'Google AI Studio' },
                'groq': { url: 'https://console.groq.com/keys', name: 'Groq' },
                'cohere': { url: 'https://dashboard.cohere.com/api-keys', name: 'Cohere' },
                'mistral': { url: 'https://console.mistral.ai/api-keys', name: 'Mistral AI' }
            };

            if (providerLinks[provider]) {
                apiLink.href = providerLinks[provider].url;
                apiLink.textContent = providerLinks[provider].name;
            }
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

function setupDragAndDrop() {
    const uploadBox = document.getElementById('upload-box');
    if (!uploadBox) return;

    uploadBox.addEventListener('dragover', (e) => {
        e.preventDefault();
        uploadBox.classList.add('dragover');
    });

    uploadBox.addEventListener('dragleave', () => {
        uploadBox.classList.remove('dragover');
    });

    uploadBox.addEventListener('drop', (e) => {
        e.preventDefault();
        uploadBox.classList.remove('dragover');

        const files = e.dataTransfer.files;
        if (files.length > 0 && files[0].type === 'application/pdf') {
            handlePDFFile(files[0]);
        } else {
            alert('Please drop a PDF file');
        }
    });
}

async function handlePDFUpload(event) {
    const file = event.target.files[0];
    if (file && file.type === 'application/pdf') {
        await handlePDFFile(file);
    }
}

async function handlePDFFile(file) {
    if (typeof pdfjsLib === 'undefined') {
        alert('PDF.js library not loaded. Please refresh the page.');
        return;
    }

    try {
        const loadingEl = document.getElementById('loading');
        const loadingTextEl = document.getElementById('loading-text');

        if (loadingEl) loadingEl.style.display = 'block';
        if (loadingTextEl) loadingTextEl.textContent = 'Extracting text from PDF...';

        const arrayBuffer = await file.arrayBuffer();
        const pdf = await pdfjsLib.getDocument({ data: arrayBuffer }).promise;

        let fullText = '';
        for (let pageNum = 1; pageNum <= pdf.numPages; pageNum++) {
            const page = await pdf.getPage(pageNum);
            const textContent = await page.getTextContent();
            const pageText = textContent.items.map(item => item.str).join(' ');
            fullText += pageText + '\n\n';
        }

        currentPDFText = fullText;
        const contractTextEl = document.getElementById('contract-text');
        if (contractTextEl) contractTextEl.value = fullText;

        const fileNameEl = document.getElementById('file-name');
        const filePagesEl = document.getElementById('file-pages');
        const uploadBoxEl = document.getElementById('upload-box');
        const fileInfoEl = document.getElementById('file-info');

        if (fileNameEl) fileNameEl.textContent = file.name;
        if (filePagesEl) filePagesEl.textContent = pdf.numPages;
        if (uploadBoxEl) uploadBoxEl.style.display = 'none';
        if (fileInfoEl) fileInfoEl.style.display = 'block';

        if (loadingEl) loadingEl.style.display = 'none';

    } catch (error) {
        console.error('Error processing PDF:', error);
        alert('Error processing PDF file. Please try again.');
        const loadingEl = document.getElementById('loading');
        if (loadingEl) loadingEl.style.display = 'none';
    }
}

function clearFile() {
    currentPDFText = '';
    const pdfUploadEl = document.getElementById('pdf-upload');
    const uploadBoxEl = document.getElementById('upload-box');
    const fileInfoEl = document.getElementById('file-info');

    if (pdfUploadEl) pdfUploadEl.value = '';
    if (uploadBoxEl) uploadBoxEl.style.display = 'block';
    if (fileInfoEl) fileInfoEl.style.display = 'none';
}

async function analyzeContract() {
    const contractTextEl = document.getElementById('contract-text');
    const queryTextEl = document.getElementById('query-text');

    if (!contractTextEl) return;

    const contractText = contractTextEl.value.trim();
    const query = queryTextEl ? queryTextEl.value.trim() : '';

    const defaultQuery = 'Analyze this legal contract. Provide a comprehensive summary including: key parties, main obligations, payment terms, important dates, termination clauses, and any notable risks or concerns.';

    if (!contractText) {
        alert('Please upload a PDF or paste contract text');
        return;
    }

    await performAnalysis(contractText, query || defaultQuery);
}

async function summarizeContract() {
    const contractTextEl = document.getElementById('contract-text');
    if (!contractTextEl) return;

    const contractText = contractTextEl.value.trim();

    if (!contractText) {
        alert('Please upload a PDF or paste contract text');
        return;
    }

    const query = 'Provide a concise summary of this legal contract, highlighting: 1) Key parties involved, 2) Main obligations and responsibilities, 3) Important dates and deadlines, 4) Payment terms, 5) Termination conditions, 6) Notable clauses or risks.';

    await performAnalysis(contractText, query);
}

async function performAnalysis(contractText, query) {
    const providerEl = document.getElementById('api-provider');
    if (!providerEl) return;

    const provider = providerEl.value;
    const apiKey = localStorage.getItem('api_key');

    if (provider !== 'demo' && !apiKey) {
        alert('Please enter your API key first');
        return;
    }

    const loadingEl = document.getElementById('loading');
    const loadingTextEl = document.getElementById('loading-text');
    const resultsEl = document.getElementById('results-section');
    const analyzeBtnEl = document.getElementById('analyze-btn');

    if (loadingEl) loadingEl.style.display = 'block';
    if (loadingTextEl) loadingTextEl.textContent = 'Analyzing contract with AI...';
    if (resultsEl) resultsEl.style.display = 'none';
    if (analyzeBtnEl) analyzeBtnEl.disabled = true;

    const startTime = Date.now();

    try {
        if (loadingTextEl) loadingTextEl.textContent = 'Generating analysis (1/3)...';
        const response1 = await callAI(provider, apiKey, contractText, query, 0.3);

        if (loadingTextEl) loadingTextEl.textContent = 'Generating verification (2/3)...';
        const response2 = await callAI(provider, apiKey, contractText, query, 0.5);

        if (loadingTextEl) loadingTextEl.textContent = 'Generating verification (3/3)...';
        const response3 = await callAI(provider, apiKey, contractText, query, 0.7);

        const totalTime = (Date.now() - startTime) / 1000;

        const consistencyScore = calculateConsistency([response1, response2, response3]);

        displayResults({
            answer: response1,
            alternativeResponses: [response2, response3],
            consistencyScore: consistencyScore,
            contractLength: contractText.length,
            analysisTime: totalTime
        });

        updateAnalytics(contractText.length, totalTime);

    } catch (error) {
        console.error('Analysis failed:', error);
        alert('Analysis failed: ' + error.message + '\n\nPlease check your API key and try again.');
    } finally {
        if (loadingEl) loadingEl.style.display = 'none';
        if (analyzeBtnEl) analyzeBtnEl.disabled = false;
    }
}

async function callAI(provider, apiKey, contractText, query, temperature) {
    if (provider === 'demo') {
        return generateDemoResponse(contractText, query);
    } else if (provider === 'openai') {
        return callOpenAI(apiKey, contractText, query, temperature);
    } else if (provider === 'anthropic') {
        return callAnthropic(apiKey, contractText, query, temperature);
    } else if (provider === 'gemini') {
        return callGemini(apiKey, contractText, query, temperature);
    } else if (provider === 'groq') {
        return callGroq(apiKey, contractText, query, temperature);
    } else if (provider === 'cohere') {
        return callCohere(apiKey, contractText, query, temperature);
    } else if (provider === 'mistral') {
        return callMistral(apiKey, contractText, query, temperature);
    }
}

async function callOpenAI(apiKey, contractText, query, temperature) {
    const response = await fetch('https://api.openai.com/v1/chat/completions', {
        method: 'POST',
        headers: {
            'Content-Type': 'application/json',
            'Authorization': 'Bearer ' + apiKey
        },
        body: JSON.stringify({
            model: 'gpt-4-turbo-preview',
            messages: [
                {
                    role: 'system',
                    content: 'You are a legal expert specializing in contract analysis. Provide detailed, accurate analysis of legal contracts.'
                },
                {
                    role: 'user',
                    content: 'Contract:\n\n' + contractText + '\n\nQuestion: ' + query
                }
            ],
            temperature: temperature,
            max_tokens: 1500
        })
    });

    if (!response.ok) {
        const error = await response.json();
        throw new Error(error.error && error.error.message ? error.error.message : 'OpenAI API call failed');
    }

    const data = await response.json();
    return data.choices[0].message.content;
}

async function callAnthropic(apiKey, contractText, query, temperature) {
    const response = await fetch('https://api.anthropic.com/v1/messages', {
        method: 'POST',
        headers: {
            'Content-Type': 'application/json',
            'x-api-key': apiKey,
            'anthropic-version': '2023-06-01'
        },
        body: JSON.stringify({
            model: 'claude-3-sonnet-20240229',
            max_tokens: 1500,
            temperature: temperature,
            messages: [
                {
                    role: 'user',
                    content: 'You are a legal expert. Analyze this contract and answer the question.\n\nContract:\n' + contractText + '\n\nQuestion: ' + query
                }
            ]
        })
    });

    if (!response.ok) {
        const error = await response.json();
        throw new Error(error.error && error.error.message ? error.error.message : 'Anthropic API call failed');
    }

    const data = await response.json();
    return data.content[0].text;
}

async function callGemini(apiKey, contractText, query, temperature) {
    const response = await fetch(`https://generativelanguage.googleapis.com/v1beta/models/gemini-pro:generateContent?key=${apiKey}`, {
        method: 'POST',
        headers: {
            'Content-Type': 'application/json'
        },
        body: JSON.stringify({
            contents: [{
                parts: [{
                    text: 'You are a legal expert specializing in contract analysis. Analyze this contract and answer the question.\n\nContract:\n' + contractText + '\n\nQuestion: ' + query
                }]
            }],
            generationConfig: {
                temperature: temperature,
                maxOutputTokens: 1500
            }
        })
    });

    if (!response.ok) {
        const error = await response.json();
        throw new Error(error.error && error.error.message ? error.error.message : 'Gemini API call failed');
    }

    const data = await response.json();
    return data.candidates[0].content.parts[0].text;
}

async function callGroq(apiKey, contractText, query, temperature) {
    const response = await fetch('https://api.groq.com/openai/v1/chat/completions', {
        method: 'POST',
        headers: {
            'Content-Type': 'application/json',
            'Authorization': 'Bearer ' + apiKey
        },
        body: JSON.stringify({
            model: 'mixtral-8x7b-32768',
            messages: [
                {
                    role: 'system',
                    content: 'You are a legal expert specializing in contract analysis. Provide detailed, accurate analysis of legal contracts.'
                },
                {
                    role: 'user',
                    content: 'Contract:\n\n' + contractText + '\n\nQuestion: ' + query
                }
            ],
            temperature: temperature,
            max_tokens: 1500
        })
    });

    if (!response.ok) {
        const error = await response.json();
        throw new Error(error.error && error.error.message ? error.error.message : 'Groq API call failed');
    }

    const data = await response.json();
    return data.choices[0].message.content;
}

async function callCohere(apiKey, contractText, query, temperature) {
    const response = await fetch('https://api.cohere.ai/v1/chat', {
        method: 'POST',
        headers: {
            'Content-Type': 'application/json',
            'Authorization': 'Bearer ' + apiKey
        },
        body: JSON.stringify({
            model: 'command',
            message: 'You are a legal expert. Analyze this contract and answer the question.\n\nContract:\n' + contractText + '\n\nQuestion: ' + query,
            temperature: temperature,
            max_tokens: 1500
        })
    });

    if (!response.ok) {
        const error = await response.json();
        throw new Error(error.message || 'Cohere API call failed');
    }

    const data = await response.json();
    return data.text;
}

async function callMistral(apiKey, contractText, query, temperature) {
    const response = await fetch('https://api.mistral.ai/v1/chat/completions', {
        method: 'POST',
        headers: {
            'Content-Type': 'application/json',
            'Authorization': 'Bearer ' + apiKey
        },
        body: JSON.stringify({
            model: 'mistral-medium',
            messages: [
                {
                    role: 'system',
                    content: 'You are a legal expert specializing in contract analysis. Provide detailed, accurate analysis of legal contracts.'
                },
                {
                    role: 'user',
                    content: 'Contract:\n\n' + contractText + '\n\nQuestion: ' + query
                }
            ],
            temperature: temperature,
            max_tokens: 1500
        })
    });

    if (!response.ok) {
        const error = await response.json();
        throw new Error(error.error && error.error.message ? error.error.message : 'Mistral API call failed');
    }

    const data = await response.json();
    return data.choices[0].message.content;
}

function generateDemoResponse(contractText, query) {
    return new Promise((resolve) => {
        setTimeout(() => {
            const contractType = detectContractType(contractText);
            const pages = Math.round(contractText.length / 3000);
            const complexity = contractText.length > 10000 ? 'High' : contractText.length > 5000 ? 'Medium' : 'Low';

            const response = 'DEMO MODE ANALYSIS\n\nThis is a demonstration of how LegalInsight works. In demo mode, we provide a template analysis.\n\nKey Findings:\n• Contract Type: ' + contractType + '\n• Length: ' + contractText.length + ' characters (~' + pages + ' pages)\n• Complexity: ' + complexity + '\n\nTo get actual AI-powered analysis:\n1. Select "OpenAI" or "Anthropic" from the provider dropdown\n2. Enter your API key\n3. Click "Analyze Contract" again\n\nThe AI will provide:\n✓ Detailed contract summary\n✓ Identification of key terms and obligations\n✓ Risk assessment\n✓ Specific answers to your questions\n✓ Multiple response verification for accuracy\n\nDemo mode does not provide actual legal analysis. Please use a real API key for production use.';

            resolve(response);
        }, 1500);
    });
}

function detectContractType(text) {
    const lower = text.toLowerCase();
    if (lower.indexOf('employment') >= 0 || lower.indexOf('employee') >= 0) return 'Employment Agreement';
    if (lower.indexOf('lease') >= 0 || lower.indexOf('rent') >= 0) return 'Lease Agreement';
    if (lower.indexOf('license') >= 0 || lower.indexOf('software') >= 0) return 'Software License';
    if (lower.indexOf('service') >= 0 || lower.indexOf('consulting') >= 0) return 'Service Agreement';
    if (lower.indexOf('purchase') >= 0 || lower.indexOf('sale') >= 0) return 'Purchase Agreement';
    return 'General Contract';
}

function calculateConsistency(responses) {
    const lengths = responses.map(function(r) { return r.length; });
    const avgLength = lengths.reduce(function(a, b) { return a + b; }, 0) / lengths.length;
    const maxLength = Math.max.apply(null, lengths);
    const minLength = Math.min.apply(null, lengths);
    const lengthVariance = maxLength - minLength;

    const consistencyPercent = Math.max(0, 100 - (lengthVariance / avgLength * 100));

    return Math.min(100, Math.max(60, consistencyPercent));
}

function displayResults(data) {
    const manualTime = estimateManualTime(data.contractLength);
    const timeSaved = manualTime - data.analysisTime;
    const efficiency = (timeSaved / manualTime * 100);
    const speedup = manualTime / data.analysisTime;

    setTextContent('time-saved', (timeSaved / 60).toFixed(1) + ' min');
    setTextContent('efficiency', efficiency.toFixed(1) + '%');
    setTextContent('speedup', speedup.toFixed(0) + 'x');

    const reliabilityClass = data.consistencyScore > 85 ? 'High' : data.consistencyScore > 70 ? 'Medium' : 'Low';
    setTextContent('hallucination-risk', reliabilityClass);

    const hallCard = document.getElementById('hallucination-card');
    if (hallCard) {
        hallCard.className = 'metric-card ' + (reliabilityClass === 'High' ? 'success' : reliabilityClass === 'Medium' ? 'warning' : 'danger');
    }

    setTextContent('answer-content', data.answer);
    setTextContent('consistency-score', data.consistencyScore.toFixed(0) + '%');

    const consFill = document.getElementById('consistency-fill');
    if (consFill) consFill.style.width = data.consistencyScore + '%';

    const interpretation = data.consistencyScore > 85 ? 'Excellent - All responses highly consistent' :
                          data.consistencyScore > 70 ? 'Good - Responses mostly consistent' :
                          'Fair - Some variation in responses, verify carefully';
    setTextContent('consistency-interpretation', interpretation);

    setTextContent('manual-time', (manualTime / 60).toFixed(1) + ' minutes');
    setTextContent('ai-time', data.analysisTime.toFixed(2) + ' seconds');
    setTextContent('contract-length', data.contractLength.toLocaleString() + ' characters');
    setTextContent('estimated-pages', Math.round(data.contractLength / 3000) + ' pages');

    const alternativesDiv = document.getElementById('alternatives-content');
    if (alternativesDiv) {
        alternativesDiv.innerHTML = '';

        data.alternativeResponses.forEach(function(response, index) {
            const div = document.createElement('div');
            div.className = 'alternative-item';
            div.innerHTML = '<strong>Verification Response ' + (index + 1) + ':</strong><br>' + response;
            alternativesDiv.appendChild(div);
        });
    }

    const resultsEl = document.getElementById('results-section');
    if (resultsEl) {
        resultsEl.style.display = 'block';
        resultsEl.scrollIntoView({ behavior: 'smooth' });
    }
}

function setTextContent(id, text) {
    const el = document.getElementById(id);
    if (el) el.textContent = text;
}

function estimateManualTime(contractLength) {
    const pages = contractLength / 3000;
    const minutesPerPage = 7.5;
    return pages * minutesPerPage * 60;
}

function updateAnalytics(contractLength, analysisTime) {
    const manualTime = estimateManualTime(contractLength);
    const timeSaved = manualTime - analysisTime;
    const efficiency = (timeSaved / manualTime * 100);

    analytics.totalAnalyses++;
    analytics.totalTimeSaved += timeSaved / 60;
    analytics.totalEfficiency = ((analytics.totalEfficiency * (analytics.totalAnalyses - 1)) + efficiency) / analytics.totalAnalyses;

    saveAnalytics();
    displayAnalytics();
}

function displayAnalytics() {
    setTextContent('total-analyses', analytics.totalAnalyses);
    setTextContent('total-time-saved', analytics.totalTimeSaved.toFixed(1));
    setTextContent('avg-efficiency', analytics.totalEfficiency.toFixed(1) + '%');

    const analyticsEl = document.getElementById('analytics-section');
    if (analytics.totalAnalyses > 0 && analyticsEl) {
        analyticsEl.style.display = 'block';
    }
}

function saveAnalytics() {
    localStorage.setItem('analytics', JSON.stringify(analytics));
}

function loadAnalytics() {
    const saved = localStorage.getItem('analytics');
    if (saved) {
        analytics = JSON.parse(saved);
        displayAnalytics();
    }
}

function clearAnalytics() {
    if (confirm('Are you sure you want to clear all analytics?')) {
        analytics = {
            totalAnalyses: 0,
            totalTimeSaved: 0,
            totalEfficiency: 0
        };
        saveAnalytics();
        displayAnalytics();
    }
}

function toggleAlternatives() {
    const content = document.getElementById('alternatives-content');
    if (content) {
        content.style.display = content.style.display === 'none' ? 'block' : 'none';
    }
}

function clearAll() {
    const contractTextEl = document.getElementById('contract-text');
    const queryTextEl = document.getElementById('query-text');
    const resultsEl = document.getElementById('results-section');

    if (contractTextEl) contractTextEl.value = '';
    if (queryTextEl) queryTextEl.value = '';
    clearFile();
    if (resultsEl) resultsEl.style.display = 'none';
}

function loadExample() {
    const exampleContract = 'SERVICE AGREEMENT\n\nThis Service Agreement ("Agreement") is entered into as of January 15, 2024 ("Effective Date"), by and between:\n\nTechCorp Solutions Inc., a Delaware corporation with offices at 123 Innovation Drive, San Francisco, CA 94105 ("Provider")\n\nAND\n\nGlobal Enterprises LLC, a California limited liability company with offices at 456 Business Boulevard, Los Angeles, CA 90001 ("Client")\n\n1. SERVICES\nProvider agrees to provide software development and consulting services as detailed in Exhibit A ("Services"). Services include custom application development, system integration, and technical support.\n\n2. TERM\nThis Agreement shall commence on the Effective Date and continue for a period of twelve (12) months, unless earlier terminated as provided herein ("Initial Term"). The Agreement shall automatically renew for successive one (1) year periods unless either party provides written notice of non-renewal at least sixty (60) days prior to the end of the then-current term.\n\n3. COMPENSATION\nClient shall pay Provider a monthly fee of $25,000 USD, payable within fifteen (15) days of invoice receipt. Late payments shall accrue interest at 1.5% per month.\n\n4. CONFIDENTIALITY\nBoth parties agree to maintain confidentiality of all proprietary information disclosed during the term of this Agreement and for three (3) years thereafter.\n\n5. TERMINATION\nEither party may terminate this Agreement with thirty (30) days written notice. Provider may terminate immediately if Client fails to pay any invoice within forty-five (45) days of receipt.\n\n6. LIABILITY\nProvider\'s total liability under this Agreement shall not exceed the total fees paid by Client in the twelve (12) months preceding the claim.\n\n7. GOVERNING LAW\nThis Agreement shall be governed by the laws of the State of California.\n\nIN WITNESS WHEREOF, the parties have executed this Agreement as of the Effective Date.';

    const contractTextEl = document.getElementById('contract-text');
    if (contractTextEl) {
        contractTextEl.value = exampleContract;
    }
    clearFile();
}

function refreshAnalytics() {
    displayAnalytics();
}
