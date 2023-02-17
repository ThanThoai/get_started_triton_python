import numpy as np
import time
from tritonclient.utils import *
import tritonclient.http as httpclient
from threading import Thread


def infer(input_data, timeout=0, priority=1, time_sleep=0):
    time.sleep(time_sleep)
    input_data = np.array(input_data, dtype=np.int32)
    client = httpclient.InferenceServerClient(url="localhost:8000")

    inputs = httpclient.InferInput("inputs", input_data.shape, "INT32")
    inputs.set_data_from_numpy(input_data)
    outputs = httpclient.InferRequestedOutput("outputs")
    query_response = client.infer(
        model_name="pipeline_multiversion",
        inputs=[inputs],
        outputs=[outputs],
        model_version="2",
    )
    print(query_response.get_response())
    outs = query_response.as_numpy("outputs")
    print(outs)


if __name__ == "__main__":
    t1 = Thread(
        target=infer,
        args=([[1]], 1000, 1, 2),
    )
    t2 = Thread(target=infer, args=([[2]], 1000))
    t1.start()
    t2.start()
    t1.join()
    t2.join()
