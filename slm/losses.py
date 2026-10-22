# Copyright 2026 Google LLC
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#    http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
# ==============================================================================
"""Custom Keras loss function for handling masked padding tokens."""

from typing import Any

import jax
import keras


@keras.saving.register_keras_serializable(package="slm")
class CustomMaskPadLoss(keras.losses.Loss):
    """Computes SparseCategoricalCrossentropy while ignoring padding token ID."""

    def __init__(self, pad_token_id: int | None = 0, **kwargs: Any):
        super().__init__(name="custom_mask_pad_loss", **kwargs)
        self.pad_token_id = pad_token_id

    def call(self, y_true: jax.Array, y_pred: jax.Array) -> jax.Array:
        loss_fn = keras.losses.SparseCategoricalCrossentropy(
            from_logits=True,
            ignore_class=self.pad_token_id,
            reduction="sum_over_batch_size",
        )
        return loss_fn(y_true, y_pred)

    def get_config(self) -> dict[str, Any]:
        config = super().get_config()
        config.update({"pad_token_id": self.pad_token_id})
        return config
