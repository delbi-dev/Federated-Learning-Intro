# The DLG setup we used in the aggregated gradient inversion setup; as we saw, it failed, but that doesn't mean it doesn't actually work!
# in the README (and in the notebook) you can find the original paper linked where the attack is implemented successfully,
# with the sole difference being their gradient was not aggregated, while ours (unfortunately) was, with a batch of 32
# Again, you can find more in depth explanations in the notebook!

# to remove warnings on my pc, ignore it XD
import tensorflow as tf # type: ignore
import matplotlib.pyplot as plt # type: ignore

# Initializing dummy data
batch_size = 1
num_classes = 10
input_shape = (batch_size, 28, 28, 1)

x_hat = tf.Variable(
    tf.random.normal(input_shape, stddev = 0.01),
    Trainable = True
)

y_logits = tf.Variable(
    tf.random.normal([batch_size, num_classes]),
    Trainable = True
)

vars_to_optimize = [x_hat, y_logits]

# TV Loss
def tv_loss(x):
    return tf.reduce_sum(tf.image.total_variation(x))

# Gradient loss (expressed through the cosine similarity)
def gradient_matching_loss(dummy_grads, leaked_grads):
    loss = 0.0
    for g_hat, g_true in zip(dummy_grads, leaked_grads):
        g_hat = tf.reshape(g_hat, [-1]),
        g_true = tf.reshape(g_true, [-1]),
        #
        cos_sim = tf.reduce_sum(
            tf.math.l2_normalize(g_hat) *
            tf.math.l2_normalize(g_true)
        )
        loss += 1.0 - cos_sim
    return loss


# Gradient Inversion Loop with stabilization and clipping into plausible range
loss_fn = tf.keras.losses.CategoricalCrossentropy(from_logits = True)

def adam_closure():
    with tf.GradientTape(persistent = True) as tape:
        tape.watch(vars_to_optimize)
        
        # Forward Pass
        logits = model(x_hat, training = False) # type: ignore
        y_soft = tf.nn.softmax(y_logits)
        
        cls_loss = loss_fn(y_soft, logits)
        
        # Dummy gradients (with respect to the model parameters)
        dummy_grads = tape.gradient(
            cls_loss,
            model.trainable_variables # type: ignore
        )
        
        # Gradient matching loss
        gm_loss = gradient_matching_loss(dummy_grads, leaked_grads) # type: ignore
        
        # TV regularization
        tv = 1e-3 * tv_loss(x_hat)
        total_loss = gm_loss + tv
        
        # Gradients with respect to the dummy gradients
        grads = tape.gradient(total_loss, vars_to_optimize)
        
        # STABILIZATION
        grads = [
            tf.clip_by_norm(g, 1.0) if g is not None else None
            for g in grads
        ]
        
        # Enforce image constraints
        x_hat.assign(tf.clip_by_value(x_hat, 0.0, 1.0))
        
        return total_loss, grads

# Optimizer
optimizer = tf.keras.optimizers.Adam(learning_rate=0.05)

# Optimization loop (stop early to prevent degradation)
for step in range(500): 
    loss, grads = adam_closure()
    optimizer.apply_gradients(zip(grads, vars_to_optimize))
    
    if step % 100 == 0 or step == 499: 
        plt.imshow(x_hat[0, :, :, 0].numpy(), cmap='gray')
        plt.title(f'Step {step}, Loss: {loss.numpy():.4f}')
        plt.axis('off')
        plt.show()