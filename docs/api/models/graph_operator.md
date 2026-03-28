# Graph Operator Models API

`deepuq.models.graph_operator` contains grid-as-graph neural operators for
scientific fields. The public `GraphNeuralOperator2D` accepts channels-last
regular-grid tensors, converts them to a local graph internally, and ends in a
final `nn.Linear` head so last-layer Laplace remains compatible.

Typical shape contract:

- input: `[batch, height, width, channels]`
- output: `[batch, height, width, out_channels]`

The model appends normalized coordinates `(x, y)` to each node feature and uses
radius-based local neighborhoods to pass messages.

::: deepuq.models.graph_operator
