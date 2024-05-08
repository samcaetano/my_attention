from utils import *

if __name__ == '__main__':
    # input docs
    inputs = [
        ["o gato sentou-se no tapete <PAD>"],
    ]

    outputs = [
        ["the cat sit on the mat"],
    ]

    # Build embeddings
    input_embedding = embedding(inputs)
    output_embedding = embedding(outputs)

    # Build positional embeddings
    inputs_positional_encoding = positional_embeddings(
        inputs=inputs,
        type='inputs',
    )
    outputs_positional_encoding = positional_embeddings(
        inputs=outputs,
        type='outputs',
    )

    # build final encodings: embeddings + positional encodings
    final_input_embeddings = input_embedding + inputs_positional_encoding
    final_input_embeddings[:, -1, :] = 0 # because of hardcoded PAD

    fig = px.imshow(final_input_embeddings[0, :, :], y=inputs[0][0].split())
    fig.write_html('images/final_inputs_encoding.html')

    final_output_embeddings = output_embedding + outputs_positional_encoding
    fig = px.imshow(final_output_embeddings[0, :, :], y=outputs[0][0].split())
    fig.write_html('images/final_outputs_encoding.html')

    print('input', inputs)
    print('inputs:', input_embedding.shape, inputs_positional_encoding.shape, final_input_embeddings.shape)
    print()
    print(outputs)
    print('outputs:', output_embedding.shape, outputs_positional_encoding.shape, final_output_embeddings.shape)

    attention_weights = scaled_dot_attention(
        query   = final_input_embeddings[0],
        key     = final_input_embeddings[0],
        value   = final_output_embeddings[0],
    )
    fig = px.imshow(
        attention_weights[:, :],
        x=inputs[0][0].split(),
        y=outputs[0][0].split()
    )
    fig.write_html('images/attention_weights_similarity.html')

    print(attention_weights)