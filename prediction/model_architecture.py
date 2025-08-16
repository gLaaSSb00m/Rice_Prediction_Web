import tensorflow as tf
from tensorflow.keras.layers import Input, Conv2D, LayerNormalization, Dense, Flatten
from tensorflow.keras.models import Model

def create_model():
    inputs = Input(shape=(224, 224, 3), name='input_1')
    
    # Normalization Layer
    x = tf.keras.layers.Normalization(mean=[123.675, 116.28, 103.53], variance=[3409.976, 3262.694, 3291.891])(inputs)
    
    # ConvNeXt Base Stem
    x = Conv2D(128, kernel_size=(4, 4), strides=(4, 4), padding='valid', activation='linear', name='convnext_base_stem_conv')(x)
    x = LayerNormalization(name='convnext_base_stem_layernorm')(x)
    
    # Add additional layers based on the architecture from the JSON
    # (This is a simplified version; you would need to add all layers as per the JSON structure)
    
    # Final layers
    x = Flatten()(x)
    outputs = Dense(20, activation='softmax', name='dense')(x)
    
    model = Model(inputs=inputs, outputs=outputs, name='model')
    return model

# Create the model
model = create_model()
model.summary()
