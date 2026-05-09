from flask import Flask, request, jsonify, render_template, Response
from flask_cors import CORS
import sys
import json
from pathlib import Path

ROOT = Path(__file__).resolve().parent
sys.path.append(str(ROOT / "src"))

from rag.qa import ask_with_rag, ask_with_rag_stream, ask_with_rag_hybrid

app = Flask(__name__)
CORS(app)

@app.route('/')
def index():
    return render_template('chat.html')

@app.route('/api/chat', methods=['POST'])
def chat():
    try:
        data = request.json
        question = data.get('question', '')
        mode = data.get('mode', 'hybrid')  # 默认使用混合检索
        top_k = data.get('top_k', 3)
        
        if not question:
            return jsonify({'error': '请输入问题'}), 400
        
        # 根据模式选择不同的检索方式
        if mode == 'hybrid':
            result = ask_with_rag_hybrid(question, top_k=top_k)
        else:
            result = ask_with_rag(question, top_k=top_k, retrieval_mode=mode)
        
        return jsonify({
            'answer': result.get('answer', ''),
            'retrieval_mode': result.get('retrieval_mode', ''),
            'suggested_questions': result.get('suggested_questions', [])[:3]
        })
    
    except Exception as e:
        return jsonify({'error': str(e)}), 500

@app.route('/api/chat/stream', methods=['POST'])
def chat_stream():
    try:
        data = request.json
        question = data.get('question', '')
        mode = data.get('mode', 'hybrid')  # 默认使用混合检索
        top_k = data.get('top_k', 3)
        
        if not question:
            return jsonify({'error': '请输入问题'}), 400
        
        def generate():
            for chunk in ask_with_rag_stream(question, top_k=top_k, retrieval_mode=mode):
                yield f"data: {json.dumps(chunk, ensure_ascii=False)}\n\n"
        
        return Response(generate(), mimetype='text/event-stream')
    
    except Exception as e:
        return jsonify({'error': str(e)}), 500

@app.route('/api/modes', methods=['GET'])
def get_modes():
    return jsonify(['text', 'embedding', 'hybrid'])

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=5000, debug=True, threaded=True)
