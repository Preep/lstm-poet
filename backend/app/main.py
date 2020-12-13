from handler.handler import NeuralBackendHandler
from flask import Flask, request, jsonify

app = Flask(__name__)

handler = NeuralBackendHandler(
    'http://34.140.198.103:8501/v1/models/neural_poet:predict'
)


@app.route('/write_poem', methods=['POST'])
def write_poem():
    if not request.is_json:
        return jsonify({
            'error': 'Please provide JSON with "user_string" key'
        }), 422

    content = request.get_json()
    try:
        user_string = content['user_string']
    except KeyError as e:
        print(e)
        return jsonify({
            'error': 'Please provide a correct JSON with "user_string" key'
        }), 422

    try:
        poem = handler.predict_on_string(user_string)
        poem = handler.poetize(poem)
    except:
        return jsonify({
            'error': 'NeuralNetwork backend is unavailible'
        }), 422

    return jsonify({
        'poem': poem
    }), 200


if __name__ == '__main__':
    app.run(host='0.0.0.0', debug=True, port=80)
