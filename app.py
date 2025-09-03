from flask import Flask, jsonify, request, send_from_directory
from flask_cors import CORS
# from tools.qa_retrieval import llm_qa
from tools.qa_retrieval_v2 import llm_qa

app = Flask(__name__)
CORS(app)

app.config['APPLICATION_ROOT'] = 'hr-api' #告訴flask應用程式的根路徑是'/hr-api' => prefix:/hr-api
app.config['DOCUMENT_FOLDER'] = 'static/HR_docs'  # 設定文件存放的資料夾

@app.route('/api' ,methods=['GET'])
def index():
    try:
        question = request.args.get('question')
        model = request.args.get('model')
        if not question:
            return jsonify({
                'status': 400,
                'message': "Question parameter is required",
                'result': [],
                'success': False
            }), 400
        response = llm_qa(question, model)
        print(response)
        return jsonify({
            'status': 200,
            'message': "success",
            'result': response,
            'success': True
        }), 200
    except Exception as e:
        return jsonify({
            'status': 400,
            'message': e,
            'result': [],
            'success': False
        }), 400

@app.route('/api/docs/<filename>', methods=['GET'])
def get_docs(filename):
    try:
        return send_from_directory(app.config['DOCUMENT_FOLDER'], filename, as_attachment=True)
    except  Exception as e:
        return jsonify({
                'status': 404,
                'error': "檔案不存在",
                'success': False
            }), 404

@app.route('/api/test' ,methods=['GET'])
def test():
    return jsonify({
        'status': 200,
        'message': "meeting_system-api|Test endpoint is working",
        'result': [],
        'success': True
    }), 200

if __name__ == '__main__':
    app.run(debug=True, port=5555)