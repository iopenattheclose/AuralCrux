import base64
import io
import numpy as np
import requests
import torch.nn as nn
import torchaudio.transforms as T
import torch
from pydantic import BaseModel
import soundfile as sf
import librosa #changing sample rate in an audio file

from model import AudioCNN



#this class takes in an audio and transforms it into a Mel-Spectrogram

class AudioProcessor:
    def __init__(self):
        self.transform = nn.Sequential(
            T.MelSpectrogram(
                sample_rate=22050,
                n_fft=1024,
                hop_length=512,
                n_mels=128,
                f_min=0,
                f_max=11025
            ),
            T.AmplitudeToDB()
        )

    def process_audio_chunk(self, audio_data):
        waveform = torch.from_numpy(audio_data).float()

        waveform = waveform.unsqueeze(0)#adding channel dimension

        spectrogram = self.transform(waveform)

        return spectrogram.unsqueeze(0)

class InferenceRequest(BaseModel):
    audio_data: str

class AudioClassifier:
    def load_model(self):
        print("Loading models on enter")
        self.device = torch.device(
            'cuda' if torch.cuda.is_available() else 'cpu')

        model_directory = "aural_crux/artifacts/models"

        checkpoint = torch.load(f'{model_directory}/best_model.pth',
                                map_location=self.device)
        self.classes = checkpoint['classes']

        self.model = AudioCNN(num_classes=len(self.classes))
        self.model.load_state_dict(checkpoint['model_state_dict'])
        self.model.to(self.device)
        self.model.eval()

        self.audio_processor = AudioProcessor()
        print("Model loaded on enter")

    def inference(self, request: InferenceRequest):
            #deployed model in prod -> upload to s3 and then download
            #here - send file directly to inference endpoint
            
            # audio_bytes = base64.b64decode(request.audio_data)

            audio_bytes = base64.b64decode(request)

            audio_data, sample_rate = sf.read(
                io.BytesIO(audio_bytes), dtype="float32")

            if audio_data.ndim > 1:
                audio_data = np.mean(audio_data, axis=1)

            if sample_rate != 44100:
                audio_data = librosa.resample(
                    y=audio_data, orig_sr=sample_rate, target_sr=44100)

            spectrogram = self.audio_processor.process_audio_chunk(audio_data)
            spectrogram = spectrogram.to(self.device)

            with torch.no_grad():
                output, feature_maps = self.model(
                    spectrogram, return_feature_maps=True)

                output = torch.nan_to_num(output)
                probabilities = torch.softmax(output, dim=1)
                top3_probs, top3_indicies = torch.topk(probabilities[0], 3)

                predictions = [{"class": self.classes[idx.item()], "confidence": prob.item()}
                            for prob, idx in zip(top3_probs, top3_indicies)]

                viz_data = {}
                for name, tensor in feature_maps.items():
                    if tensor.dim() == 4:  # [batch_size, channels, height, width]
                        aggregated_tensor = torch.mean(tensor, dim=1)
                        squeezed_tensor = aggregated_tensor.squeeze(0)
                        numpy_array = squeezed_tensor.cpu().numpy()
                        clean_array = np.nan_to_num(numpy_array)
                        viz_data[name] = {
                            "shape": list(clean_array.shape),
                            "values": clean_array.tolist()
                        }

                spectrogram_np = spectrogram.squeeze(0).squeeze(0).cpu().numpy()
                clean_spectrogram = np.nan_to_num(spectrogram_np)

                max_samples = 8000
                waveform_sample_rate = 44100
                if len(audio_data) > max_samples:
                    step = len(audio_data) // max_samples
                    waveform_data = audio_data[::step]
                else:
                    waveform_data = audio_data

            response = {
                "predictions": predictions,
                "visualization": viz_data,
                "input_spectrogram": {
                    "shape": list(clean_spectrogram.shape),
                    "values": clean_spectrogram.tolist()
                },
                "waveform": {
                    "values": waveform_data.tolist(),
                    "sample_rate": waveform_sample_rate,
                    "duration": len(audio_data) / waveform_sample_rate
                }
            }

            return response


def predict():
    # random_audio = np.random.choice()
    audio_data, sample_rate = sf.read("aural_crux/artifacts/audio/5-244327-A-34.wav")

    buffer = io.BytesIO()
    sf.write(buffer, audio_data, sample_rate, format="WAV")
    audio_b64 = base64.b64encode(buffer.getvalue()).decode("utf-8")
    payload = {"audio_data": audio_b64}

    server = AudioClassifier()

    #this is for inference in terminal
    # server.load_model()
    # response = server.inference(audio_b64)
    # result = response


    server = AudioClassifier()
    url = server.inference.get_web_url()
    response = requests.post(url, json=payload)
    response.raise_for_status()

    result = response.json()


    waveform_info = result.get("waveform", {})
    if waveform_info:
        values = waveform_info.get("values", {})
        print(f"First 10 values: {[round(v, 4) for v in values[:10]]}...")
        print(f"Duration: {waveform_info.get("duration", 0)}")

    print("Top predictions:")
    for pred in result.get("predictions", []):
        print(f"  -{pred["class"]} {pred["confidence"]:0.2%}")


if __name__=="__main__":
    predict() 