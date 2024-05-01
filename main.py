from utils import *
import plotly.express as px

if __name__ == '__main__':
    # input docs
    inputs = [
        ["o gato sentou-se no tapete"],
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
    fig = px.imshow(final_input_embeddings[0, :, :], y=inputs[0][0].split())
    fig.write_html('final_inputs_encoding.html')

    final_output_embeddings = output_embedding + outputs_positional_encoding
    fig = px.imshow(final_output_embeddings[0, :, :], y=outputs[0][0].split())
    fig.write_html('final_outputs_encoding.html')

    print('inputs:', input_embedding.shape, inputs_positional_encoding.shape, final_input_embeddings.shape)
    print('outputs:', output_embedding.shape, outputs_positional_encoding.shape, final_output_embeddings.shape)