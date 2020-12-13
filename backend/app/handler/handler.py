import youtokentome as yttm
import requests
import os
import numpy as np


class NeuralBackendHandler():

    def __init__(self, api_url):
        self.path = os.path.dirname(os.path.abspath(__file__))
        self.api_url = api_url
        self.bpe = yttm.BPE(model=os.path.join(self.path, 'yttm.model'))
        self.vocab_size = self.bpe.vocab_size()
        self.sequence_lenght = 20
        self.newline_token = 88

    def pad_sequence(self, sequence, maxlen=20):
        if len(sequence) >= 20:
            padded_result = sequence[-20:]
        else:
            padded_result = [0]*(20-len(sequence)) + sequence
        return padded_result

    def predict_on_string(self,
                          input_string,
                          max_len=60,
                          top_k=2,
                          variant_newline=True,
                          newline_top_k=120,
                          chaos_top_k=75,
                          chaos_rate=0.2):

        s = input_string.lower().replace('ё', 'е') + ' | '
        s = self.bpe.encode(s, output_type=yttm.OutputType.ID)
        s = [self.pad_sequence(s)]
        s = np.array(s)
        chaos_vector = np.random.random_sample((max_len,))

        is_new_line = True
        result = []
        for i in range(max_len):

            payload = s.tolist()
            response = requests.post(
                self.api_url, json={'instances': payload})
            next_token_prob_space = np.array(response.json()['predictions'])

            if np.argmax(next_token_prob_space) == self.newline_token:
                next_token_sample = [self.newline_token]
                is_new_line = True
            else:
                is_chaos = chaos_vector[i] < chaos_rate and chaos_top_k != 0
                if variant_newline and is_new_line and newline_top_k != 0:
                    top_k_from = -(newline_top_k)
                    is_new_line = False
                elif is_chaos:
                    top_k_from = -(top_k + chaos_top_k)
                else:
                    top_k_from = -(top_k)

            next_token_sample = np.argpartition(
                next_token_prob_space, top_k_from
                )[0][top_k_from:]

            next_token = np.random.choice(next_token_sample)
            s = np.expand_dims(
                np.append(s, next_token)[-self.sequence_lenght:], axis=0
                )
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
