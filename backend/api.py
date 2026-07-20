"""
LegalInsight Backend API
Self-RAG + EigenScore for Legal Contract Analysis with Time Tracking
"""
import os
import sys
import time
import json
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Optional
from flask import Flask, request, jsonify
from flask_cors import CORS
import yaml

sys.path.insert(0, str(Path(__file__).parent.parent))

from src.self_rag.gguf_inference import SelfRAGGGUFInference, compute_eigenscore
from src.self_rag.self_healing_graph import build_self_healing_pipeline
from src.retrieval.retriever import LegalRetriever
from src.retrieval.embedding import EmbeddingModel
from src.retrieval.chunking import DocumentChunker
from src.guardrails import InputGuardrails, OutputGuardrails, PolicyEngine

app = Flask(__name__)
CORS(app)  # Enable CORS for GitHub Pages frontend

model = None
retriever = None
analytics_data = []

input_guardrails = InputGuardrails()
output_guardrails = OutputGuardrails()
policy_engine = PolicyEngine.from_yaml(
    str(Path(__file__).parent.parent / "configs" / "guardrails_policy.yaml")
)

class TimeTracker:
    """Track time metrics for contract analysis"""

    @staticmethod
    def estimate_manual_time(contract_length: int) -> float:
        """
        Estimate manual contract analysis time in seconds.
        Rule of thumb: 1 page (~500 words) = 5-10 minutes for basic review.
        """
        pages = contract_length / 3000  # ~3000 chars per page
        minutes = pages * 7.5  # Average of 5-10 minutes per page
        return minutes * 60

    @staticmethod
    def calculate_time_saved(actual_time: float, contract_length: int) -> Dict:
        """Calculate time savings metrics"""
        manual_time = TimeTracker.estimate_manual_time(contract_length)
        time_saved = manual_time - actual_time
        percentage_saved = (time_saved / manual_time * 100) if manual_time > 0 else 0

        return {
            "manual_analysis_time_seconds": round(manual_time, 2),
            "ai_analysis_time_seconds": round(actual_time, 2),
            "time_saved_seconds": round(time_saved, 2),
            "time_saved_minutes": round(time_saved / 60, 2),
            "efficiency_improvement_percent": round(percentage_saved, 2),
            "speedup_factor": round(manual_time / actual_time, 2) if actual_time > 0 else 0
        }

def initialize_model():
    """Initialize the Self-RAG model"""
    global model

    model_path = os.getenv("SELFRAG_MODEL_PATH", "data/models/selfrag-7b-q4_k_m.gguf")

    if not os.path.exists(model_path):
        return {
            "error": "Model not found",
            "message": f"Please download the Self-RAG model to {model_path}",
            "download_url": "https://huggingface.co/selfrag/selfrag_llama2_7b"
        }

    try:
        model = SelfRAGGGUFInference(
            model_path=model_path,
            n_ctx=4096,
            n_gpu_layers=0,  # Set to -1 for GPU acceleration
            verbose=False
        )
        return {"status": "Model initialized successfully"}
    except Exception as e:
        return {"error": f"Failed to initialize model: {str(e)}"}

def initialize_retriever():
    """Initialize the retrieval system"""
    global retriever

    try:
        config_path = Path(__file__).parent.parent / "configs" / "retrieval_config.yaml"
        with open(config_path, 'r') as f:
            config = yaml.safe_load(f)

        embedding_model = EmbeddingModel(
            model_name=config['embedding']['model_name'],
            device=config['embedding'].get('device', 'cpu')
        )

        chunker = DocumentChunker(
            chunk_size=config['chunking']['chunk_size'],
            chunk_overlap=config['chunking']['chunk_overlap']
        )

        retriever = LegalRetriever(
            embedding_model=embedding_model,
            chunker=chunker,
            top_k=config['retrieval']['top_k']
        )

        return {"status": "Retriever initialized successfully"}
    except Exception as e:
        return {"error": f"Failed to initialize retriever: {str(e)}"}

@app.route('/health', methods=['GET'])
def health_check():
    """Health check endpoint"""
    return jsonify({
        "status": "healthy",
        "model_loaded": model is not None,
        "retriever_loaded": retriever is not None
    })

@app.route('/initialize', methods=['POST'])
def initialize():
    """Initialize the model and retriever"""
    model_result = initialize_model()
    retriever_result = initialize_retriever()

    return jsonify({
        "model": model_result,
        "retriever": retriever_result
    })

@app.route('/analyze_contract', methods=['POST'])
def analyze_contract():
    """
    Analyze a legal contract using Self-RAG + EigenScore
    """
    start_time = time.time()

    try:
        data = request.json
        contract_text = data.get('contract_text', '')
        query = data.get('query', 'Summarize this contract and identify key terms, obligations, and risks.')
        use_retrieval = data.get('use_retrieval', True)
        num_generations = data.get('num_generations', 5)  # For EigenScore

        if not contract_text:
            return jsonify({"error": "No contract text provided"}), 400

        if model is None:
            return jsonify({"error": "Model not initialized. Call /initialize first"}), 503

        input_check = input_guardrails.check_query(query)
        input_policy = policy_engine.check(query, applies_to="input")
        if not input_check.allowed or not input_policy.allowed:
            return jsonify({
                "error": "Query blocked by guardrails",
                "reasons": input_check.blocked_reasons + [v.reason for v in input_policy.violations],
            }), 400

        retrieval_context = None
        if use_retrieval and retriever is not None:
            retrieval_start = time.time()
            documents = [{"text": contract_text, "metadata": {"source": "user_contract"}}]
            retriever.index_documents(documents)

            results = retriever.retrieve(query, top_k=3)
            retrieval_context = "\n\n".join([r['text'] for r in results])
            retrieval_time = time.time() - retrieval_start
        else:
            retrieval_time = 0

        generation_start = time.time()

        responses = []
        issup_tokens = []
        for i in range(num_generations):
            passage = retrieval_context if (use_retrieval and retrieval_context) else None
            output = model.generate(
                question=query,
                passage=passage,
                max_tokens=512,
                temperature=0.7 if i > 0 else 0.1,  # First one more deterministic
            )
            responses.append(output.answer)
            issup_tokens.append(output.issup)

        eigenscore = None
        if retriever is not None and retriever.embedding_model is not None:
            eigenscore = compute_eigenscore(responses, retriever.embedding_model)[0]

        generation_time = time.time() - generation_start
        total_time = time.time() - start_time

        output_check = output_guardrails.check_answer(responses[0])
        output_policy = policy_engine.check(responses[0], applies_to="output")
        guardrail_blocked = not output_check.allowed or not output_policy.allowed
        if guardrail_blocked:
            responses[0] = (
                "This response was withheld by output guardrails "
                f"({', '.join(output_check.blocked_reasons + [v.reason for v in output_policy.violations])})."
            )

        time_metrics = TimeTracker.calculate_time_saved(total_time, len(contract_text))

        analytics_entry = {
            "timestamp": datetime.now().isoformat(),
            "contract_length": len(contract_text),
            "query": query,
            "total_time": total_time,
            "retrieval_time": retrieval_time,
            "generation_time": generation_time,
            "eigenscore": eigenscore,
            "time_metrics": time_metrics
        }
        analytics_data.append(analytics_entry)

        if eigenscore is None:
            hallucination_risk = "Unknown"
        elif eigenscore < -2.0:
            hallucination_risk = "Low"
        elif eigenscore < 0:
            hallucination_risk = "Medium"
        else:
            hallucination_risk = "High"

        response = {
            "answer": responses[0],  # Primary response
            "alternative_responses": responses[1:],
            "eigenscore": eigenscore,
            "hallucination_risk": hallucination_risk,
            "issup_tokens": issup_tokens,
            "guardrails": {
                "input_redacted_query": input_check.redacted_text if input_check.pii_findings else None,
                "output_blocked": guardrail_blocked,
                "output_blocked_reasons": (
                    output_check.blocked_reasons + [v.reason for v in output_policy.violations]
                    if guardrail_blocked else []
                ),
            },
            "time_metrics": time_metrics,
            "performance": {
                "total_time_seconds": round(total_time, 2),
                "retrieval_time_seconds": round(retrieval_time, 2),
                "generation_time_seconds": round(generation_time, 2)
            },
            "contract_stats": {
                "length_characters": len(contract_text),
                "estimated_pages": round(len(contract_text) / 3000, 1)
            }
        }

        return jsonify(response)

    except Exception as e:
        return jsonify({"error": str(e)}), 500

@app.route('/analyze_contract_self_healing', methods=['POST'])
def analyze_contract_self_healing():
    """
    Analyze a contract using the self-healing RAG loop: retrieve -> generate
    -> critique -> (on rejection) reformulate query and re-retrieve, up to
    max_attempts, before falling back to an "insufficient information"
    response instead of a fabricated answer.
    """
    try:
        data = request.json
        contract_text = data.get('contract_text', '')
        query = data.get('query', 'Summarize this contract and identify key terms, obligations, and risks.')
        max_attempts = data.get('max_attempts', 2)

        if not contract_text:
            return jsonify({"error": "No contract text provided"}), 400
        if model is None or retriever is None:
            return jsonify({"error": "Model/retriever not initialized. Call /initialize first"}), 503

        input_check = input_guardrails.check_query(query)
        input_policy = policy_engine.check(query, applies_to="input")
        if not input_check.allowed or not input_policy.allowed:
            return jsonify({
                "error": "Query blocked by guardrails",
                "reasons": input_check.blocked_reasons + [v.reason for v in input_policy.violations],
            }), 400

        start_time = time.time()
        retriever.index_documents(
            [{"text": contract_text, "metadata": {"source": "user_contract"}}]
        )

        pipeline = build_self_healing_pipeline(model, retriever, max_attempts=max_attempts)
        state = pipeline.run(query)
        total_time = time.time() - start_time

        answer = state["result"].answer
        output_check = output_guardrails.check_answer(answer)
        output_policy = policy_engine.check(answer, applies_to="output")
        guardrail_blocked = not output_check.allowed or not output_policy.allowed
        if guardrail_blocked:
            answer = (
                "This response was withheld by output guardrails "
                f"({', '.join(output_check.blocked_reasons + [v.reason for v in output_policy.violations])})."
            )

        return jsonify({
            "answer": answer,
            "used_fallback": state["used_fallback"],
            "attempts": len(state["trace"]),
            "trace": state["trace"],
            "guardrails": {
                "output_blocked": guardrail_blocked,
                "output_blocked_reasons": (
                    output_check.blocked_reasons + [v.reason for v in output_policy.violations]
                    if guardrail_blocked else []
                ),
            },
            "performance": {"total_time_seconds": round(total_time, 2)},
        })

    except Exception as e:
        return jsonify({"error": str(e)}), 500

@app.route('/summarize_contract', methods=['POST'])
def summarize_contract():
    """Quick contract summarization endpoint"""
    try:
        data = request.json
        contract_text = data.get('contract_text', '')

        return analyze_contract_internal(
            contract_text=contract_text,
            query="Provide a concise summary of this contract, highlighting key parties, main obligations, important dates, and notable clauses."
        )
    except Exception as e:
        return jsonify({"error": str(e)}), 500

@app.route('/answer_query', methods=['POST'])
def answer_query():
    """Answer specific questions about a contract"""
    try:
        data = request.json
        contract_text = data.get('contract_text', '')
        query = data.get('query', '')

        if not query:
            return jsonify({"error": "No query provided"}), 400

        return analyze_contract_internal(
            contract_text=contract_text,
            query=query
        )
    except Exception as e:
        return jsonify({"error": str(e)}), 500

@app.route('/analytics', methods=['GET'])
def get_analytics():
    """Get analytics data"""
    total_analyses = len(analytics_data)
    if total_analyses == 0:
        return jsonify({
            "total_analyses": 0,
            "total_time_saved_minutes": 0,
            "average_efficiency_improvement": 0
        })

    total_time_saved = sum(a['time_metrics']['time_saved_seconds'] for a in analytics_data)
    avg_efficiency = sum(a['time_metrics']['efficiency_improvement_percent'] for a in analytics_data) / total_analyses

    return jsonify({
        "total_analyses": total_analyses,
        "total_time_saved_minutes": round(total_time_saved / 60, 2),
        "total_time_saved_hours": round(total_time_saved / 3600, 2),
        "average_efficiency_improvement_percent": round(avg_efficiency, 2),
        "recent_analyses": analytics_data[-10:]
    })

def analyze_contract_internal(contract_text: str, query: str):
    """Internal helper for contract analysis"""
    # Mutates request.json so analyze_contract() can be called directly, since it reads from the request
    request.json = {
        'contract_text': contract_text,
        'query': query,
        'use_retrieval': True,
        'num_generations': 5
    }
    return analyze_contract()

if __name__ == '__main__':
    print("LegalInsight API Server")
    print("=" * 50)
    print("Initializing model and retriever...")

    model_result = initialize_model()
    retriever_result = initialize_retriever()

    print(f"Model: {model_result}")
    print(f"Retriever: {retriever_result}")
    print("=" * 50)
    print("Server starting on http://localhost:5000")

    app.run(host='0.0.0.0', port=5000, debug=True)
