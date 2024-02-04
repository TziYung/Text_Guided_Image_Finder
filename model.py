import tensorflow as tf

class CLIP(tf.keras.Model):
    """
    The model take a text encoder and image encoder and try to project it into 
    same dimentional space. By taking pairs of image and text, the output of 
    encoders is projected through its own projector. After that, dot product is
    applied, creating a matix of [batch_size, batch_size]. The matrix contains 
    similarties of each image and text. For example, the [0, 1] would be how 
    similar the first text and the second image is.
    To evalute the performance of model, cross entropy is used to optimzed the 
    similiarity of the correct pairs in the matrix. First it calculate the loss
    of each text against all the images in the batch. Then it calculate the 
    loss of each image against all the text in the batch.
    """
    def __init__(self, text_encoder, image_encoder, dim, image_encoder_head):
        super().__init__()
        self.text_encoder = text_encoder
        self.image_encoder = image_encoder
        self.attention_pooling = AttentionPooling(image_encoder_head, dim)
        self.text_projector = Projector(dim)
        self.image_projector = Projector(dim)
        self.similarity = Similarity()

        self.loss_tracker = tf.keras.metrics.Mean(name = "Loss")

        self.ln = tf.keras.layers.LayerNormalization()

    def call(self, input_, training = False):
        # Quick note, the performance of CLIP is less sensitve in capcacity of text encoder
       
        input_id, attention_mask, image = input_
        t_f = self.text_encoder(input_id, attention_mask, training = training).last_hidden_state
        t_f = self.ln(t_f)
        # take the EOT of each text, since it considered to contains most information
        # 102 is the end id
        eot = tf.where(tf.equal(input_id, 102))
        
        t_f = tf.gather_nd(t_f, eot)
        
        t_e = self.text_projector(t_f)

        i_f = self.image_encoder(image, training = training)
        i_f = self.attention_pooling(i_f)
        i_f = tf.keras.layers.Flatten()(i_f)
        i_e = self.image_projector(i_f)

        return self.similarity(t_e, i_e)

    def train_step(self, input_):
        # In here only the input data is used
        input_ = input_[0]
        with tf.GradientTape() as tape:
            tape.watch(self.trainable_variables)
            matrix = self(input_, training = True)
            loss = self.loss_fn(matrix)
        trainable_vars = self.trainable_variables
        gradient = tape.gradient(loss, trainable_vars)
        self.opt.apply_gradients(zip(gradient, trainable_vars))
        self.loss_tracker.update_state(loss)

        return {tracker.name: tracker.result() for tracker in self.metrics}
    def compile(self, opt):
        super().compile()
        self.opt = opt
    @property
    def metrics(self):
        return [self.loss_tracker]

    def loss_fn(self, similarity):
        labels = tf.range(tf.shape(similarity)[0], dtype = "float32")
        t_loss = tf.keras.losses.SparseCategoricalCrossentropy(
            from_logits = True,
        )(labels, similarity)
        t_loss = tf.reduce_mean(t_loss)
        i_loss = tf.keras.losses.SparseCategoricalCrossentropy(
            from_logits = True,
        )(labels, tf.transpose(similarity))
        i_loss = tf.reduce_mean(i_loss)
        return (t_loss + i_loss) / 2
     
class Projector(tf.keras.layers.Layer):
    """
    The class project the output of text encoder or image encoder to the same dimension.
    Actually it is just a fully connected layer with l2_normalize packed into a class so 
    that it could be used by both image encoder and text encoder.
    """
    def __init__(self, dim):
        super().__init__()
        self.dim = dim

    def build(self, input_shape):
        self.w = self.add_weight(
            shape = (input_shape[-1], self.dim),
            initializer = "random_normal",
            trainable = True,
        )
        self.b = self.add_weight(
            shape = (self.dim,),
            initializer = "random_normal",
            trainable = True,
        )
    def call(self, inputs):
        emb = tf.matmul(inputs, self.w) + self.b
        emb = tf.math.l2_normalize(emb)

        return emb

class Similarity(tf.keras.layers.Layer):
    def __init__(self):
        super().__init__()
        # Initial should be 0.07 and no larger than 100
        """
        The reason to apply temerature after dot product
        ''
        During contrastive learning, we interpret the dot products 
        as logits that are fed to the cross-entropy loss for each 
        text and image example. Because we normalize the feature 
        vectors, the dot products are capped between [-1, 1], which
        might not have an enough dynamic range and limit the categorical
        probability distributions that the logits can express. For this
        reason, we scale the logits to have larger differences. To avoid
        numerical instability, we clipped the scale value at 100, and
        for all models that we trained, it reached 100
        ''
        """
        
        self.temperature = self.add_weight(
            shape = (1,),
            # This use the default value in CLIP research paper
            initializer = tf.keras.initializers.Constant(value = 0.07),
            # The clip research paper state that for the stability, the temperature should be <= 100
            constraint = ConstraintLEQ(),
            trainable = True,
        ) 

    def call(self, text_emb, image_emb):
        # Same as tf.matmul
        similarity = tf.einsum("ax, bx -> ab",text_emb, image_emb) 
        similarity *= tf.math.exp(self.temperature) 
        return similarity
class AttentionPooling(tf.keras.layers.Layer):
    def __init__(self, num_head, key_dim):
        super().__init__()
        self.num_head = num_head
        self.key_dim = key_dim
        self.attention = tf.keras.layers.MultiHeadAttention(self.num_head, self.key_dim)
        self.positional_encoding = PositionalEncoding()
        
    def call(self, inputs):
        shape = inputs.shape
        layer = tf.keras.layers.Reshape((-1, shape[-1]))(inputs) # Change the shape to N(H*W)C


        mean_ = tf.reduce_mean(layer, axis = 1, keepdims = True)


        # It is replace by reduce mean

        layer = tf.keras.layers.Concatenate(axis = 1)([mean_, layer]) # N, (H * W) + 1, C

        layer = self.positional_encoding(layer)
        
        mean_ = layer[:, :1, :] # Only use the mean as the query of attention

        layer = self.attention(mean_, layer)

 
        return layer
class PositionalEncoding(tf.keras.layers.Layer):
    def __init__(self):
        super().__init__()
    def build(self, input_shape):
        self.pe = self.add_weight(
            shape = ( input_shape),
            trainable = True,
        )    
    def call(self, inputs):
        return self.pe + inputs

class ConstraintLEQ(tf.keras.constraints.Constraint):
    def __call__(self, w):
        # If less than 100 it would be w * 1 + 100.0 * 0
        # If larger than 100 it would be w * 0 + 100.0 * 1
        return w * tf.cast(tf.math.less_equal(w, 100.), w.dtype) + 100. * tf.cast(tf.math.greater(w, 100.), w.dtype)
