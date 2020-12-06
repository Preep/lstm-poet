import tensorflow as tf
from tensorflow import keras
import youtokentome as yttm
import numpy as np
import os

os.environ["CUDA_VISIBLE_DEVICES"]="-1" 

class NeuralPoet():

    def __init__(self):
        self.path = os.path.dirname(os.path.abspath(__file__))
        self.model = keras.models.load_model(os.path.join(self.path, 'lstm_nlp.h5'))
        self.bpe = yttm.BPE(model=os.path.join(self.path, 'yttm.model'))
        self.vocab_size = self.bpe.vocab_size()
        self.sequence_lenght = 20
        self.newline_token = 88

    def predict_on_string(self,
                          input_string,
                          max_len=60,
                          top_k=2,
                          variant_newline=True,
                          newline_top_k=100,
                          chaos_top_k=15,
                          chaos_rate=0.25):

        s = input_string.lower().replace('ё', 'е') + ' | '
        s = self.bpe.encode(s, output_type=yttm.OutputType.ID)
        s = tf.keras.preprocessing.sequence.pad_sequences([s], maxlen=self.sequence_lenght)
        chaos_vector = np.random.random_sample((max_len,))

        is_new_line = True
        result = []
        for i in range(max_len):

            next_token_prob_space = self.model.predict(s)
            if np.argmax(next_token_prob_space) == self.newline_token:
                next_token_sample = [self.newline_token]
                is_new_line = True
            else:
                is_chaos = (chaos_vector[i] < chaos_rate) and (chaos_top_k != 0)
                if variant_newline and is_new_line and newline_top_k != 0:
                    top_k_from = -(newline_top_k)
                    is_new_line = False
                elif is_chaos:
                    top_k_from = -(top_k + chaos_top_k)
                else:
                    top_k_from = -(top_k)

            next_token_sample = np.argpartition(next_token_prob_space, top_k_from)[0][top_k_from:]

            next_token = np.random.choice(next_token_sample)
            s = np.expand_dims(np.append(s, next_token)[-self.sequence_lenght:], axis=0)
            result.append(next_token)

        predicted_string = self.bpe.decode([result])[0]
        final_string = input_string + ' | ' + predicted_string
        final_string = final_string.replace(' | ', '\n').replace('|', '')
        return final_string


    def poetize(self, input_string):
        lines_list = input_string.split('\n')
        len_list = list(map(len, lines_list))
        if (sum(len_list[:-1])/len(len_list[:-1])) > (len_list[-1] * 0.7):
            lines_list = lines_list[:-1]

        result = []
        for line in lines_list:
            result.append(line.capitalize())
        return '\n'.join(result)


if __name__ == '__main__':
    import sys
    input_string = sys.argv[1]
    poet = NeuralPoet()
    neural_poem = poet.predict_on_string(input_string)
    neural_poem = poet.poetize(neural_poem)
    print(neural_poem)
