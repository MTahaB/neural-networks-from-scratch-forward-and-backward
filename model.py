"""
Neural Networks From Scratch: Forward and Backward

Assembled from your step-by-step solutions.
"""

import numpy as np

# Step 1 - numerical_gradient
def numerical_gradient(f, x, eps=1e-5):
    x = np.array(x, dtype=float, copy=True)
    grad = np.zeros_like(x)

    for idx in np.ndindex(x.shape):
        original = x[idx]

        x[idx] = original + eps
        f_plus = f(x)

        x[idx] = original - eps
        f_minus = f(x)

        x[idx] = original
        grad[idx] = (f_plus - f_minus) / (2 * eps)

    return grad

# Step 2 - gradient_check
def gradient_check(analytic_grad, numeric_grad, tol=1e-5):
    analytic_grad = np.asarray(analytic_grad, dtype=float)
    numeric_grad = np.asarray(numeric_grad, dtype=float)

    diff = np.abs(analytic_grad - numeric_grad)
    scale = np.maximum(
        np.maximum(np.abs(analytic_grad), np.abs(numeric_grad)),
        tol
    )

    return float(np.max(diff / scale, initial=0.0))

# Step 3 - make_dense
def make_dense(in_dim, out_dim, weight_init_fn):
    """Create a fully connected layer.

    Inputs:
      in_dim: int, input feature size
      out_dim: int, output feature size
      weight_init_fn: callable(in_dim, out_dim) -> (W, b)

    Returns layer dict with keys:
      params: {'W': (in_dim, out_dim), 'b': (out_dim,)}
      forward(x) -> (y, cache) with y shape (batch, out_dim)
      backward(dout, cache) -> (dx, grads) with grads {'W', 'b'}
        Analytic dx/dW/db must match numerical_gradient via gradient_check.
    """
    W, b = weight_init_fn(in_dim, out_dim)
    params = {'W': W, 'b': b}

    def forward(x):
        y = x @ params['W'] + params['b']
        return y, x

    def backward(dout, cache):
        x = cache

        dx = dout @ params['W'].T
        grads = {
            'W': x.T @ dout,
            'b': dout.sum(axis=0),
        }
        return dx, grads

    return {
        'params': params,
        'forward': forward,
        'backward': backward,
    }

# Step 4 - make_activation
def make_activation(kind='relu'):
    """Create a genuinely nonlinear elementwise activation layer.

    Args:
        kind: str nonlinearity name. Default 'relu' must implement ReLU
              (zero negatives, pass non-negatives). Other kinds optional.

    Returns:
        Layer dict with:
          forward(x) -> (y, cache)
            x, y: np.ndarray shape (batch, dim)
          backward(dout, cache) -> (dx, {})
            dout, dx: np.ndarray shape (batch, dim)
            param grad dict is always empty (no learnable params)

    Must be elementwise and non-affine; analytic dx must match
    numerical_gradient / gradient_check.
    """
    def forward(x):
        y = np.maximum(0, x)
        cache = x > 0
        return y, cache

    def backward(dout, cache):
        dx = dout * cache
        return dx, {}

    return {
        'params': {},
        'forward': forward,
        'backward': backward,
    }

# Step 5 - initialize_weights
def initialize_weights(in_dim, out_dim, scheme='he'):
    """Return (W, b) for a dense layer.

    Inputs:
      in_dim: int fan-in
      out_dim: int fan-out
      scheme: str initialization family (default 'he')

    Returns:
      W: np.ndarray shape (in_dim, out_dim), finite, symmetry-breaking,
         scale stable with depth (fan-in dependent)
      b: np.ndarray shape (out_dim,), near zero
    """
    if scheme == 'he':
        scale = np.sqrt(2.0 / in_dim)


    W = np.random.randn(in_dim, out_dim) * scale
    b = np.zeros(out_dim, dtype=float)

    return W, b

# Step 6 - make_loss
def make_loss(kind='cross_entropy'):
    """Return a classification loss_fn(logits, labels) -> (loss, d_logits).

    Inputs to loss_fn:
      logits: (batch, C) float array of raw class scores
      labels: (batch,) int array of class indices in [0, C)
    Outputs:
      loss: Python float, mean scalar loss over the batch (finite)
      d_logits: (batch, C) gradient of loss w.r.t. logits (finite)
    Must pass gradient_check, be minimized by confident correct predictions,
    and stay finite under saturated logits.
    """
    def loss_fn(logits, labels):
        logits = np.asarray(logits, dtype=float)
        labels = np.asarray(labels, dtype=int)
        batch_size = logits.shape[0]
        rows = np.arange(batch_size)

        shifted = logits - np.max(logits, axis=1, keepdims=True)
        exp_scores = np.exp(shifted)
        sums = np.sum(exp_scores, axis=1, keepdims=True)

        log_probs = shifted - np.log(sums)
        loss = float(-np.mean(log_probs[rows, labels]))

        d_logits = exp_scores / sums
        d_logits[rows, labels] -= 1.0
        d_logits /= batch_size

        return loss, d_logits

    return loss_fn

# Step 7 - make_sequential
def make_sequential(layers):
    """Compose protocol-honoring layers into one sequential model.

    Inputs:
      layers: list of layer dicts, each with
        forward(x) -> (y, cache),
        backward(dout, cache) -> (dx, grads_dict),
        params: dict of ndarrays (possibly empty).

    Returns a dict with:
      forward(x) -> (y, caches)
        y: final activation after applying every layer in order
        caches: opaque structure needed by backward
      backward(dout, caches) -> (dx, grads_list)
        dx: gradient w.r.t. the original input x
        grads_list: list of length len(layers); grads_list[i] is the
          grads_dict from layers[i] ({} for param-free layers)
      params: aggregated live view of all layer params, length len(layers),
        same order as layers (so in-place updates affect the model)
    """
    layers = list(layers)

    def forward(x):
        caches = []

        for layer in layers:
            x, cache = layer['forward'](x)
            caches.append(cache)

        return x, caches

    def backward(dout, caches):
        grads_list = [None] * len(layers)

        for i in reversed(range(len(layers))):
            dout, grads = layers[i]['backward'](dout, caches[i])
            grads_list[i] = grads

        return dout, grads_list

    return {
        'forward': forward,
        'backward': backward,
        'params': [layer['params'] for layer in layers],
    }

# Step 8 - forward_backward (not yet solved)
# TODO: implement

# Step 9 - make_optimizer (not yet solved)
# TODO: implement

# Step 10 - train_step (not yet solved)
# TODO: implement

# Step 11 - train (not yet solved)
# TODO: implement

# Step 12 - design_network (not yet solved)
# TODO: implement

# Step 13 - improve_generalization (not yet solved)
# TODO: implement

