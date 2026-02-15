# Some useful functions used in the experiment; again, you can find a better explanation on each one on the main notebook!

# gradient matching to each layer of our architecture; this is fundamental as basically each layer type (dense weights, bias, conv kernel...) encodes different informations
# this means we can't just compare a gradient from a conv kernel to one of the dense bias, else the problem becomes ill-posed
layer_gradients = {}

grad_index = 0
grad_keys = delta_data.files  # type: ignore # ['arr_0', 'arr_1', ...]

for layer in model.layers: # type: ignore
    grads_for_layer = []

    for param in layer.weights:
        grad = delta_data[grad_keys[grad_index]] # type: ignore

        if grad.shape != param.shape:
            raise ValueError(
                f"Shape mismatch at {layer.name}/{param.name}: "
                f"{grad.shape} vs {param.shape}"
            )

        grads_for_layer.append(grad)
        grad_index += 1

    if grads_for_layer:
        layer_gradients[layer.name] = grads_for_layer

print("Gradients successfully mapped to model layers:")
for k, v in layer_gradients.items():
    print(f"{k}: {[g.shape for g in v]}")

# sanity check; checks layer order, kernel and bias alignment and shape matching (for each variable var.shape must be equal to grad.shape)
for var, grad in zip(model.trainable_variables, dummy_grads): # type: ignore
    print(var.name, grad.shape)