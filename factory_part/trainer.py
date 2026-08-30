from typing import Any

from .objective import spotlight_loss


class SpotlightTrainerMixin:
    def compute_loss(self, model: Any, inputs: dict[str, Any], return_outputs: bool = False, **kwargs: Any):
        token_weights = inputs.pop("token_weights", None)
        if token_weights is None:
            return super().compute_loss(model, inputs, return_outputs=return_outputs, **kwargs)
        labels = inputs.pop("labels")
        outputs = model(**inputs)
        loss = spotlight_loss(outputs.logits, labels, token_weights)
        if return_outputs:
            return loss, outputs
        return loss
