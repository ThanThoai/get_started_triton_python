import json
import triton_python_backend_utils as pb_utils
import time
import numpy as np


class TritonPythonModel:
    def initialize(self, args):
        print("Load Model")

    def execute(self, requests):
        print("Version--2")
        responses = []
        print(len(requests))
        for request in requests:
            inp = pb_utils.get_input_tensor_by_name(request, "inputs")
            out = pb_utils.InferenceResponse(
                output_tensors=[pb_utils.Tensor("outputs", np.array(inp.as_numpy()))]
            )
            time.sleep(2)
            responses.append(out)
        return responses
