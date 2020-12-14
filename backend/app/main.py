from handler.handler import NeuralBackendHandler
from flask import Flask, request, jsonify

app = Flask(__name__)

handler = NeuralBackendHandler(
    'http://37.140.198.103:8501/v1/models/neural_poet:predict'
)


@app.route('/write_poem', methods=['POST'])
def write_poem():
    if not request.is_json:
        answer = (jsonify({
            'error': 'Please provide JSON with "user_string" key'
        }), 422
        )

    content = request.get_json()
    try:
        user_string = content['user_string']
        print(f'USER INPUT: {user_string}')
    except KeyError:
        answer = (jsonify({
            'error': 'Please provide a correct JSON with "user_string" key'
        }), 422
        )

    try:
        poem = handler.predict_on_string(user_string)
        poem = handler.poetize(poem)
    except:
        answer = (jsonify({
            'error': 'NeuralNetwork backend is unavailible'
        }), 500
        )

    answer = (jsonify({
        'poem': poem
    }), 200
    )
    print(f'ANSWER: {answer}')
    return answer


if __name__ == '__main__':
    app.run(host='0.0.0.0', debug=True, port=80)
