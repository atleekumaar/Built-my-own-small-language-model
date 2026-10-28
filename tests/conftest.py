import os

# Ensure Keras defaults to the JAX backend for all test sessions
os.environ["KERAS_BACKEND"] = "jax"
os.environ["JAX_PLATFORMS"] = "cpu"
