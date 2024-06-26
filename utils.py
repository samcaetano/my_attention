import numpy as np
import plotly.express as px

# embedding size
d_model = 6

def _softmax(x):
    exp_scores = np.exp(x)
    return exp_scores / np.sum(exp_scores, axis=1, keepdims=True)

def embedding(inputs : list) -> np.array:
    """
    Build embedding matrix for a given input.

    Args: 
        inputs: the list of documents to be embedded
    Returns:
        the embedding matrix
    """
    # number of documents
    n = len(inputs)

    # number of tokens within the documents
    m = len(inputs[0][0].split())

    E = np.random.rand(n, m, d_model)

    return E

def positional_embeddings(inputs : list, type : str) -> np.array:
    """
    Build positional encoding embeddings for a given input.
    This will encode each dimension of the embedding dimension by (same as the Attention Is All You Need paper):

    PE(pos,2i) = sin(pos/100002i/dmodel), for even-th dimensions
    PE(pos,2i+1) = cos(pos/100002i/dmodel), for odd-th dimensions

    Args:   
        inputs: the list of documents to be encoded
    Returns:        
        the positional encoding matrix
    """
    # number of documents
    n = len(inputs)

    # number of tokens within the documents
    m = len(inputs[0][0].split())

    # randomly initialize PE matrix
    PE = np.random.rand(n, m, d_model)

    for idx_doc, doc in enumerate(inputs):
        # print(doc)
        for pos, token in enumerate(doc[0].split()):            
            for i in range(0, d_model, 2):
                sin_enc = np.sin(pos / (10000**(2*i/d_model))) # apply for even i-dimensions (0th, 2nd, 4th, ..., (d_model-1)th)
                cos_enc = np.cos(pos / (10000**(((2*i)+1)/d_model))) # apply for odd i-dimensions (1st, 3rd, 5th, ..., d_model-th)

                PE[idx_doc, pos, i] = sin_enc
                PE[idx_doc, pos, i+1] = cos_enc

            # print(pos, token, PE[idx_doc, pos, :])
    # draw PEs
    fig = px.imshow(
            PE[0, :, :],
            y=doc[0].split(),
        )
    fig.write_html(f'images/{type}_positional_encoding.html')

    return PE

def scaled_dot_attention(query, key, value):
    # matmul instead of np.dot to handle matrices greater than 2-D
    # this captures the similarity of each word related to all other words in the sentence
    att_similarity = np.matmul(query, key.T)
    #print(att_similarity)

    # this captures the attention to to give to each word in the sentence
    att_weights = _softmax(att_similarity / np.sqrt(d_model))
    #print(att_weights)

    # this captures the similarity of each word attention rate to all other words in the output sentence
    return np.matmul(att_weights, value)