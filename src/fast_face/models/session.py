import os
import numpy as np
import onnxruntime as ort
from typing import List, Optional

from ..schema import ProviderType

class ONNXSession:
    def __init__(
        self,
        model_path: str = os.path.join(os.path.dirname(os.path.abspath(__file__)), "mobilenet_retinaface.onnx"),
        providers: List[ProviderType] = ["CPUExecutionProvider"],
        sess_options: Optional[ort.SessionOptions] = None
    ):
        self.session = ort.InferenceSession(
            model_path,
            providers = providers,
            sess_options=sess_options
        )
    
    def process(self, inputs:np.ndarray):
        assert len(inputs.shape) == 4, "Inputs shape length != 4"
        input_name = self.session.get_inputs()[0].name
        outputs = self.session.run(None, {input_name: inputs})
        return outputs
    
    def __call__(self, inputs: np.ndarray):
        return self.process(inputs=inputs)