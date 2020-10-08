### Introduction to TensorFlow
We are going to build a simple neural network that predicts the price of house based on a simple formulae.
Imagine that the price of the house is 50k and per bedroom 50k increase. so one bedroom house price is 100k. A simple neural network was build using tensorflow to calculate the price of the house
```python
#import tensorflow
import tensorflow as tf
import numpy as np
from tensorflow import keras

#house_model
def house_model(y_new):
    xs = np.array([1.0, 2.0, 3.0, 4.0, 5.0, 6.0], dtype=float)
    ys = np.array([1.0, 1.5, 2.0, 2.5, 3.0, 3.5], dtype=float)
    model = tf.keras.Sequential([keras.layers.Dense(units=1, input_shape=[1])])
    model.compile(optimizer='sgd', loss='mean_squared_error')
    model.fit(xs, ys, epochs=500)
    return model.predict(y_new)[0]

prediction = house_model([7.0])
print(prediction)

```