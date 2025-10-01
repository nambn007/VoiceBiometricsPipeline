from pprint import pprint

import numpy as np
from speechbrain.inference.speaker import EncoderClassifier
import torch
import torchaudio

from silero_vad import (
    collect_chunks,
    get_speech_timestamps,
    load_silero_vad,
    read_audio,
)

class VoiceBiometricsPipeline:
    def __init__(self, sampling_rate=16000) -> None:
        self.sampling_rate = sampling_rate

        # Load silero-vad 
        self.vad_model = load_silero_vad(onnx=True)

        self.get_speech_timestamps = get_speech_timestamps
        self.collect_chunks = collect_chunks
        self.read_audio = read_audio

        # Load ecapa-tdnn 
        self.speaker_model = EncoderClassifier.from_hparams(
            source="speechbrain/spkrec-ecapa-voxceleb",
            savedir="pretrained_models/spkrec-ecapa-voxceleb"
        )

        self.min_speech_duration = 0.5 # seconds 

    def extract_embedding(self, audio_path):
        """Extract speaker embedding from file audio""" 

        # Load audio 
        wav = self.read_audio(audio_path, sampling_rate=self.sampling_rate)

        # Dump wav to file "debug_wav{count}.bin" for debugging, incrementing count per call
        if not hasattr(self, "_wav_dump_count"):
            self._wav_dump_count = 0
        self._wav_dump_count += 1
        wav_np = wav.cpu().numpy() if hasattr(wav, 'cpu') else np.array(wav)
        file_path = f"debug_wav{self._wav_dump_count}.bin"
        with open(file_path, "wb") as f:
            f.write(wav_np.astype(np.float32).tobytes())
        
        # VAD --> speech timestamps 
        speech_timestamps = self.get_speech_timestamps(
            wav,
            self.vad_model,
            # sampling_rate=self.sampling_rate,
            # threshold=0.5,
            # min_speech_duration_ms=250,
            # min_silence_duration_ms=100,
            # return_seconds=True
        )
        print(speech_timestamps)

        if len(speech_timestamps) == 0:
            return None, []

        # Build list of chunks (list[Tensor]) from timestamps to preserve segments
        speech_chunks = []
        for ts in speech_timestamps:
            start, end = ts['start'], ts['end']
            chunk = wav[start:end]
            # Skip empty chunks defensively
            if chunk.numel() == 0:
                continue
            # Filter by min duration (seconds)
            if (end - start) / self.sampling_rate < self.min_speech_duration:
                continue
            speech_chunks.append((chunk, ts))

        print("Size of chunks: ", len(speech_chunks))
        count = 0
        res = []
        for temp in speech_chunks[0][0]:
            if temp > 0:
                res.append(temp.item())
                count += 1
            if count > 100:
                break
        # print(speech_chunks[0][0][:100])
        print(res)

        # Extract embedding 
        embeddings = []
        valid_chunks = []

        for chunk, ts in speech_chunks:
            if chunk.dim() == 1:
                chunk = chunk.unsqueeze(0)
            embedding = self.speaker_model.encode_batch(chunk)
            embeddings.append(embedding.squeeze().cpu())
            valid_chunks.append(ts)

        if len(embeddings) == 0:
            return None, []

        final_embedding = torch.stack(embeddings).mean(dim=0)

        return final_embedding, valid_chunks

    def extract_embedding_from_waveform(self, waveform):
        """Extract embedding from waveform tensor"""
        # VAD 
        speech_timestamps = self.get_speech_timestamps(
            waveform,
            self.vad_model,
            sampling_rate=self.sampling_rate,
            threshold=0.5
        )

        if len(speech_timestamps) == 0:
            return None

        # Collect chunks 
        speech_chunks = self.collect_chunks(speech_timestamps, waveform)
        full_speech = torch.cat(speech_chunks)

        if full_speech.dim() == 1:
            full_speech = full_speech.unsqueeze(0)

        embedding = self.speaker_model.encode_batch(full_speech)

        return embedding.squeeze().cpu()
        
    def compute_similarity(self, embedding1, embedding2):
        """Calculate cosnine similarity"""
        similarity = torch.nn.functional.cosine_similarity(
            embedding1.unsqueeze(0),
            embedding2.unsqueeze(0)
        )

        return similarity.item()
