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

sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

from self_rag.gguf_inference import SelfRAGGGUFInference, compute_eigenscore
from retrieval.retriever import LegalRetriever
from retrieval.embedding import EmbeddingModel
from retrieval.chunking import DocumentChunker

app = Flask(__name__)
CORS(app)  # Enable CORS for GitHub Pages frontend

model = None
retriever = None
analytics_data = []

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
        for i in range(num_generations):
            if use_retrieval and retrieval_context:
                output = model.generate_with_retrieval(
                    query=query,
                    retrieved_docs=retrieval_context,
                    max_new_tokens=512,
                    temperature=0.7 if i > 0 else 0.1  # First one more deterministic
                )
            else:
                output = model.generate_without_retrieval(
                    query=query,
                    max_new_tokens=512,
                    temperature=0.7 if i > 0 else 0.1
                )
            responses.append(output.answer)

        eigenscore = compute_eigenscore(responses, model.model)

        generation_time = time.time() - generation_start
        total_time = time.time() - start_time

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

        response = {
            "answer": responses[0],  # Primary response
            "alternative_responses": responses[1:],
            "eigenscore": eigenscore,
            "hallucination_risk": "Low" if eigenscore < -2.0 else "Medium" if eigenscore < 0 else "High",
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
