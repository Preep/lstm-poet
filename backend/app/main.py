from handler.handler import NeuralBackendHandler
from flask import Flask, request, jsonify

app = Flask(__name__)

handler = NeuralBackendHandler(
    'http://0.0.0.0:8501/v1/models/neural_poet:predict'
)


@app.route('/write_poem', methods=['POST'])
def write_poem():
    if not request.is_json:
        print(f'NON-JSON RECEIVED:\n{request.data}')
        return jsonify({'error': 'Please provide JSON with "user_string" key'}), 422
    content = request.get_json()

    try:
        user_string = content['user_string']
        print(f'ON USER INPUT:\n{user_string}')
    except KeyError:
        print(f'INCORRECT JSON RECEIVED:\n{request.data}')
        return jsonify({'error': 'Please provide a correct JSON with "user_string" key'}), 422

    try:
        poem = handler.predict_on_string(user_string, poetize_after_prediction=True)
    except ConnectionError as e:
        print(f'NEURAL BACKEND NOT RESPONDED:\n{e}')
        return jsonify({'error': 'NeuralNetwork backend is unavailible'}), 500

    print('GENERATED:\n', poem)
    return jsonify({'poem': poem}), 200


if __name__ == '__main__':
    app.run(host='0.0.0.0', debug=False, port=80)
