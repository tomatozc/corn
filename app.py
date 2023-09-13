from flask import Flask, request, jsonify
# from predict_animals import PredictImg_animals
from predict_corn import PredictImg_corn

app = Flask(__name__)

# @app.route('/upload', methods=['POST'])
# def upload():
#     if 'file' not in request.files:
#         return jsonify({'error': 'No file uploaded'}), 400
    
#     file = request.files['file']
#     if file.filename == '':
#         return jsonify({'error': 'No file selected'}), 400
    
#     savePath = 'testimg.jpg'
#     file.save(savePath)
    
#     predict = PredictImg_animals()
#     result = predict.onnxPredict(savePath)
#     # name = result[0]
#     # confidence = result[1]
#     # resultList = [name, confidence]

#     return jsonify({'result': result}), 200

@app.route('/upload_corn', methods=['POST'])
def upload_corn():
    if 'file' not in request.files:
        return jsonify({'error': 'No file uploaded'}), 400
    
    file = request.files['file']
    if file.filename == '':
        return jsonify({'error': 'No file selected'}), 400
    
    savePath = 'testimg.jpg'
    file.save(savePath)
    
    predict = PredictImg_corn()
    result = predict.onnxPredict(savePath)
    return jsonify({'result': result}), 200


if __name__ == '__main__':
    app.run(host='0.0.0.0')