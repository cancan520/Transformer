# %matplotlib inline
import numpy as np
import tensorflow as tf

# 设置gpu内存自增长
gpus = tf.config.experimental.list_physical_devices('GPU')
for gpu in gpus:
    tf.config.experimental.set_memory_growth(gpu, True)


# 加载训练时保存的 tokenizer
import tensorflow_datasets as tfds
# 如果是TensorFlow2.0，请将下面的"deprecatrd"换成"features"
en_tokenizer = tfds.deprecated.text.SubwordTextEncoder.load_from_file('./tokenizer/en_tokenizer_new')
pt_tokenizer = tfds.deprecated.text.SubwordTextEncoder.load_from_file('./tokenizer/en_tokenizer_new')

def positional_encoding(max_len, d_model):
    pos = np.arange(max_len,dtype=float)[:,np.newaxis]
    dim = np.arange(d_model,dtype=float)[np.newaxis,:]
    matrix = np.multiply(pos, 1 / np.power(10000,2*(dim//2)/np.float32(d_model)))
    matrix[:,::2] = np.sin(matrix[:,::2])
    matrix[:, 1::2] = np.cos(matrix[:, 1::2])
    pos_encoding = np.expand_dims(matrix, 0)
    pos_encoding = tf.cast(pos_encoding,tf.float32)
    return pos_encoding

def scaled_pot_product_attention(Q,K,V,mask=None):
    QK = tf.matmul(Q,K,transpose_b=True) # shape:(batch_size, heads_num, seq_len, seq_len)
    dk = tf.cast(tf.shape(K)[-1], tf.float32)
    scaled_attention_logits = QK / tf.math.sqrt(dk) # shape:(batch_size, heads_num, seq_len, seq_len)

    if mask is not None:
        scaled_attention_logits += tf.multiply(mask,-1e9) # shape:(batch_size, heads_num, seq_len, seq_len)

    attention_weights = tf.nn.softmax(scaled_attention_logits, axis=-1) # shape:(batch_size, heads_num, seq_len, seq_len)

    attention_output = tf.matmul(attention_weights, V)  # shape:(batch_size, heads_num, seq_len, depth)
    return attention_output, attention_weights


def FeedForward(dff, d_model):
    return tf.keras.Sequential([
                tf.keras.layers.Dense(units=dff, activation='relu'),
                tf.keras.layers.Dense(d_model),
            ])

def create_padding_mask(input_data):
    padding_mask = tf.cast(tf.math.equal(input_data,0),tf.float32)
    padding_mask = padding_mask[:,tf.newaxis,tf.newaxis,:]
    return padding_mask

def create_look_ahead_mask(input_data):
    '''
    :param input_data: shape:(batch_size, input_seq_len)  input_seq_len包含start跟end
    :return:
    '''
    seq_len = tf.shape(input_data)[1]
    return 1-tf.linalg.band_part(tf.ones((seq_len,seq_len)),-1,0)

class MultiHeadAttention(tf.keras.layers.Layer):
    def __init__(self, d_model, heads_num):
        super(MultiHeadAttention, self).__init__()
        self.WQ = tf.keras.layers.Dense(units=d_model)
        self.WK = tf.keras.layers.Dense(units=d_model)
        self.WV = tf.keras.layers.Dense(units=d_model)
        self.dense = tf.keras.layers.Dense(d_model)
        self.depth = d_model // heads_num
        self.heads_num = heads_num
        self.d_model = d_model

    def split_head(self, input):
        '''
        :param input: shape: (batch_size, seq_len, d_model)
        :return: shape: (batch_size, heads_num, seq_len, depth)
        '''
        batch_size = tf.shape(input)[0]
        seq_len = tf.shape(input)[1]
        input = tf.reshape(input, (batch_size, seq_len, self.heads_num, self.depth)) # shape:(batch_size, seq_len, heads_num, depth)
        return tf.transpose(input,perm=[0,2,1,3]) # shape:(batch_size, heads_num, seq_len, depth)

    def call(self,q,k,v,padding_mask):
        '''
        :param q:  shape:(batch_size, input_seq_len, d_model)
        :param k:  shape:(batch_size, input_seq_len, d_model)
        :param v:  shape:(batch_size, input_seq_len, d_model)
        :return:   shape:(batch_size, input_seq_len, d_model)    shape:(batch_size, heads_num, seq_len, seq_len)
        '''
        Q = self.WQ(q) # shape:(batch_size, seq_len, d_model)
        K = self.WK(k) # shape:(batch_size, seq_len, d_model)
        V = self.WV(v) # shape:(batch_size, seq_len, d_model)

        Q = self.split_head(Q) # shape:(batch_size, heads_num, seq_len, depth)
        K = self.split_head(K) # shape:(batch_size, heads_num, seq_len, depth)
        V = self.split_head(V) # shape:(batch_size, heads_num, seq_len, depth)

        attention_output, attention_weights = scaled_pot_product_attention(Q,K,V,padding_mask) # shape:(batch_size, heads_num, seq_len, depth)
        attention_output = tf.transpose(attention_output, perm=[0,2,1,3]) # shape:(batch_size, seq_len, heads_num, depth)
        batch_size = tf.shape(q)[0]
        seq_len = tf.shape(q)[1]
        concat_attention = tf.reshape(attention_output, (batch_size, seq_len, self.d_model)) # shape:(batch_size, input_seq_len, d_model)
        output = self.dense(concat_attention) # shape:(batch_size, input_seq_len, d_model)

        return output, attention_weights

def positional_encoding(max_len, d_model):
    pos = np.arange(max_len,dtype=float)[:,np.newaxis]
    dim = np.arange(d_model,dtype=float)[np.newaxis,:]
    matrix = np.multiply(pos, 1 / np.power(10000,2*(dim//2)/np.float32(d_model)))
    matrix[:,::2] = np.sin(matrix[:,::2])
    matrix[:, 1::2] = np.cos(matrix[:, 1::2])
    pos_encoding = np.expand_dims(matrix, 0)
    pos_encoding = tf.cast(pos_encoding,tf.float32)
    return pos_encoding

# *******************************************************************************
class DecoderLayer(tf.keras.layers.Layer):
    def __init__(self, d_model, heads_num, dff, rate=0.1):
        super(DecoderLayer, self).__init__()
        self.mha1 = MultiHeadAttention(d_model, heads_num)
        self.dropout1 = tf.keras.layers.Dropout(rate)
        self.layer_norm1 = tf.keras.layers.LayerNormalization(epsilon=1e-6)

        self.mha2 = MultiHeadAttention(d_model, heads_num)
        self.dropout2 = tf.keras.layers.Dropout(rate)
        self.layer_norm2 = tf.keras.layers.LayerNormalization(epsilon=1e-6)

        self.ffn = FeedForward(dff, d_model)
        self.dropout3 = tf.keras.layers.Dropout(rate)
        self.layer_norm3 = tf.keras.layers.LayerNormalization(epsilon=1e-6)

    def call(self, input, encoder_ouput, encoder_decoder_padding_mask, masked_attention_mask, training=False):
        '''
        :param input:          shape:  (batch_size, target_seq_len, d_model)
        :param encoder_ouput:  shape:  (batch_size, input_seq_len, d_model)
        :param encoder_decoder_padding_mask:
        :param masked_attention_mask:
        :return:
        '''
        q = input
        k = input
        v = input
        attn1_output, attn1_weights = self.mha1(q, k, v, masked_attention_mask)  # shape:(batch_size, target_seq_len, d_model)
        attn1_output = self.dropout1(attn1_output,training=training) # shape:(batch_size, input_seq_len, d_model)
        output1 = self.layer_norm1(input + attn1_output) # shape:(batch_size, target_seq_len, d_model)

        q = output1        # shape: (batch_size, target_seq_len, d_model)
        k = encoder_ouput  # shape: (batch_size, input_seq_len, d_model)
        v = encoder_ouput  # shape: (batch_size, input_seq_len, d_model)
        attn2_output, attn2_weights = self.mha2(q, k, v, encoder_decoder_padding_mask)  # shape:(batch_size, target_seq_len, d_model)
        attn2_output = self.dropout2(attn2_output,training=training) # shape:(batch_size, input_seq_len, d_model)
        output2 = self.layer_norm2(output1 + attn2_output) # shape:(batch_size, target_seq_len, d_model)

        ffn_output = self.ffn(output2)  # shape:(batch_size, target_seq_len, d_model)
        ffn_output = self.dropout3(ffn_output, training=training)  # shape:(batch_size, target_seq_len, d_model)
        output3 = self.layer_norm3(output2 + ffn_output)  # shape:(batch_size, target_seq_len, d_model)

        return output3

class DecoderModel(tf.keras.layers.Layer):
    def __init__(self, target_vocab_size, d_model, max_len, heads_num, dff, layers_num, rate=0.1):
        super(DecoderModel, self).__init__()
        self.d_model = d_model
        self.max_len = max_len
        self.embedding = tf.keras.layers.Embedding(target_vocab_size, d_model)
        self.decoder_layers = [DecoderLayer(d_model, heads_num, dff, rate) for _ in range(layers_num)]
        self.dropout = tf.keras.layers.Dropout(rate)

    def call(self, input, encoder_ouput, encoder_decoder_padding_mask, training=False):
        '''
        :param input:         shape:(batch_size, target_seq_len)  target_seq_len只包含start
        :param encoder_ouput: shape:(batch_size, input_seq_len, d_model)
        :param training:
        :return:
        '''
        decoder_padding_mask = create_padding_mask(input)
        look_ahead_mask = create_look_ahead_mask(input)
        masked_attention_mask = tf.maximum(decoder_padding_mask, look_ahead_mask)

        input_embedding = self.embedding(input) # shape:(batch_size, target_seq_len, d_model)
        input_embedding = input_embedding * tf.math.sqrt(tf.cast(self.d_model,tf.float32)) # shape:(batch_size, target_seq_len, d_model)
        pos_encoding = positional_encoding(self.max_len, self.d_model)  # shape: (1, max_len, d_model)

        input_seq_len = tf.shape(input)[1]
        input_pos_embedding = input_embedding + pos_encoding[:,:input_seq_len,:] # shape:(batch_size, input_seq_len, d_model)
        x = self.dropout(input_pos_embedding, training=training) # shape:(batch_size, input_seq_len, d_model)

        for decoder_layer in self.decoder_layers:
            x = decoder_layer(x, encoder_ouput, encoder_decoder_padding_mask, masked_attention_mask, training=training) # shape:(batch_size, input_seq_len, d_model)
        decoder_ouput = x
        return decoder_ouput # shape:(batch_size, input_seq_len, d_model)
# *******************************************************************************






# *******************************************************************************
class EncoderLayer(tf.keras.layers.Layer):
    def __init__(self, d_model, heads_num, dff, rate=0.1):
        super(EncoderLayer, self).__init__()
        self.mha = MultiHeadAttention(d_model, heads_num)
        self.dropout1 = tf.keras.layers.Dropout(rate)
        self.layer_norm1 = tf.keras.layers.LayerNormalization(epsilon=1e-6)
        self.ffn = FeedForward(dff, d_model)
        self.dropout2 = tf.keras.layers.Dropout(rate)
        self.layer_norm2 = tf.keras.layers.LayerNormalization(epsilon=1e-6)

    def call(self, input, padding_mask, training=False):
        '''
        :param input:  shape: (batch_size, input_seq_len, d_model)
        :param training:
        :return:
        '''
        q = input
        k = input
        v = input
        attn_output, attn_weights = self.mha(q,k,v,padding_mask) # shape:(batch_size, input_seq_len, d_model)
        attn_output = self.dropout1(attn_output,training=training) # shape:(batch_size, input_seq_len, d_model)
        output1 = self.layer_norm1(input + attn_output) # shape:(batch_size, input_seq_len, d_model)

        ffn_output = self.ffn(output1) # shape:(batch_size, input_seq_len, d_model)
        ffn_output = self.dropout2(ffn_output,training=training) # shape:(batch_size, input_seq_len, d_model)
        output2 = self.layer_norm2(output1 + ffn_output) # shape:(batch_size, input_seq_len, d_model)

        return output2 # shape:(batch_size, input_seq_len, d_model)

class EncoderModel(tf.keras.layers.Layer):
    def __init__(self, input_vocab_size, d_model, max_len, heads_num, dff, layers_num, rate=0.1):
        super(EncoderModel, self).__init__()
        self.d_model = d_model
        self.max_len = max_len
        self.embedding = tf.keras.layers.Embedding(input_vocab_size, d_model)
        self.encoder_layers = [EncoderLayer(d_model, heads_num, dff, rate) for _ in range(layers_num)]
        self.dropout = tf.keras.layers.Dropout(rate)

    def call(self, input, training=False):
        '''
        :param input:  shape:(batch_size, input_seq_len)  input_seq_len包含start跟end
        :param training:
        :return:
        '''
        padding_mask = create_padding_mask(input)

        input_embedding = self.embedding(input) # shape:(batch_size, input_seq_len, d_model)
        input_embedding = input_embedding * tf.math.sqrt(tf.cast(self.d_model,tf.float32)) # shape:(batch_size, input_seq_len, d_model)
        pos_encoding = positional_encoding(self.max_len, self.d_model)  # shape: (1, max_len, d_model)

        input_seq_len = tf.shape(input)[1]
        input_pos_embedding = input_embedding + pos_encoding[:,:input_seq_len,:] # shape:(batch_size, input_seq_len, d_model)
        x = self.dropout(input_pos_embedding, training=training) # shape:(batch_size, input_seq_len, d_model)
        for encoder_layer in self.encoder_layers:
            x = encoder_layer(x, padding_mask, training=training) # shape:(batch_size, input_seq_len, d_model)
        encoder_ouput = x
        return encoder_ouput, padding_mask # shape:(batch_size, input_seq_len, d_model)

# *******************************************************************************

class Transformer(tf.keras.Model):
    def __init__(self, input_vocab_size, target_vocab_size, d_model, max_len, heads_num, dff, layers_num, rate=0.1):
        super(Transformer, self).__init__()
        self.EncoderModel = EncoderModel(input_vocab_size, d_model, max_len, heads_num, dff, layers_num)
        self.DecoderModel = DecoderModel(target_vocab_size, d_model, max_len, heads_num, dff, layers_num)
        self.linear = tf.keras.layers.Dense(units=target_vocab_size)


    def call(self, encoder_input, decoder_input, training=False):
        '''
        :param encoder_input:  shape:(batch_size, input_seq_len)  input_seq_len包含start跟end
        :param decoder_input:  shape:(batch_size, target_seq_len)  target_seq_len只包含end
        :param training:
        :return:
        '''
        encoder_ouput, encoder_decoder_padding_mask = self.EncoderModel(encoder_input, training=training)
        decoder_ouput = self.DecoderModel(decoder_input, encoder_ouput, encoder_decoder_padding_mask, training=training)
        predictions = self.linear(decoder_ouput) # shape:(batch_size, input_seq_len, d_model)
        predictions = tf.nn.softmax(predictions, axis=-1)
        return predictions # shape: (batch_size, target_seq_len, target_vocab_size)

input_vocab_size = pt_tokenizer.vocab_size + 2
target_vocab_size = en_tokenizer.vocab_size + 2
d_model = 128
max_len = 40
heads_num = 8
dff = 512
layers_num = 4
transformer = Transformer(input_vocab_size, target_vocab_size, d_model, max_len, heads_num, dff, layers_num)

checkpoint = tf.train.Checkpoint(model=transformer)
checkpoint.restore(tf.train.latest_checkpoint('./checkpoint'))


def evalute(inp_sentence, model):
    input_id_sentence = [pt_tokenizer.vocab_size] + pt_tokenizer.encode(inp_sentence) + [pt_tokenizer.vocab_size + 1]
    encoder_input = tf.expand_dims(input_id_sentence, 0)  # (1,input_sentence_length)
    decoder_input = tf.expand_dims([en_tokenizer.vocab_size], 0)  # (1,1)
    for i in range(max_len):
        predictions = model(encoder_input, decoder_input, training=False)
        predictions = predictions[:, -1, :]  # 单步
        predictions_id = tf.cast(tf.argmax(predictions, axis=-1), tf.int32)  # 预测概率最大的值
        if tf.equal(predictions_id, en_tokenizer.vocab_size + 1):
            return tf.squeeze(decoder_input, axis=0)

        decoder_input = tf.concat([decoder_input, [predictions_id]],
                                  axis=-1)
    return tf.squeeze(decoder_input, axis=0)


def translate(input_sentence, model):
    result = evalute(input_sentence, model)
    predicted_sentence = en_tokenizer.decode([i for i in result if i < en_tokenizer.vocab_size])
    print("Input: {}".format(input_sentence))
    print("Predicted translation: {}".format(predicted_sentence))

# # 正式使用
import sys
if __name__ == "__main__":
    if len(sys.argv)>1:
        input_sentence = sys.argv[1]
        translate(input_sentence.lower(),transformer)
    else:
        print('参数不足')