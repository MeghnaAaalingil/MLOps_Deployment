import numpy as np
import onnxruntime as ort
from scipy.special import softmax

from data import DataModule
from utils import timing


class ColaONNXPredictor:
    def __init__(self, model_path):
        # Create Inference Session which will load the onnx model
        self.ort_session = ort.InferenceSession(model_path)
        self.processor = DataModule()
        self.lables = ["unacceptable", "acceptable"]

    @timing
    def predict(self, text):
        inference_sample = {"sentence": text}
        processed = self.processor.tokenize_data(inference_sample)
        
        # Prepare the inputs for the session
        # The input names should match the names used while creating the onnx model.
        
        ort_inputs = {
            "input_ids": np.expand_dims(processed["input_ids"], axis=0),
            "attention_mask": np.expand_dims(processed["attention_mask"], axis=0),
        }

        # Run the inference session with the inputs
        ort_outs = self.ort_session.run(None, ort_inputs)
        scores = softmax(ort_outs[0])[0]
        predictions = []
        for score, label in zip(scores, self.lables):
            predictions.append({"label": label, "score": score})
        return predictions


if __name__ == "__main__":
    sentence = "The boy is sitting on a bench"
    #./models/
    predictor = ColaONNXPredictor("./models/model.onnx")
    print(predictor.predict(sentence))
    sentences = ["The boy is sitting on a bench"]*10
    for sentence in sentences:
        predictor.predict(sentence)