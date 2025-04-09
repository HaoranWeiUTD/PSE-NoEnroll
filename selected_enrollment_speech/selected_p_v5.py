# 随机挑5s, 使用sil-VAD去除静音段和纯噪音段
import numpy as np
import soundfile as sf
import os
import argparse
from silero_vad import load_silero_vad, get_speech_timestamps
import random

SAMPLING_RATE = 16000
length = 5
CHUNK_SIZE = length * 16000
model = load_silero_vad()

def vad(wav):
    speech_timestamps = get_speech_timestamps(wav, model)
    return speech_timestamps

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--wav_path", default="/Share/wsl/data/win/youtube_data_lecture/path/noisy_path.txt")
    parser.add_argument("--save_dir", default="/Share/wsl/data/win/youtube_data_lecture/enrol_selected-5s-random-speech")
    args = parser.parse_args()

    os.makedirs(args.save_dir, exist_ok=True)
    with open(args.wav_path, 'r') as f:
        for path in f:
            path = path.strip()
            audio, fs = sf.read(path)
            assert fs == SAMPLING_RATE
            speech_timestamps = vad(audio)
            random.shuffle(speech_timestamps)

            audio_seg = np.empty((0,))
            for non_speech_seg in speech_timestamps:
                length = non_speech_seg['end'] - non_speech_seg['start']
                if length > CHUNK_SIZE:   
                    idx_seg = np.random.randint(0, len(audio)-CHUNK_SIZE)
                    audio_seg = audio[idx_seg:idx_seg+CHUNK_SIZE]
                    break
                else:
                    continue
            if len(audio_seg) > 0:
                save_path = os.path.join(args.save_dir, os.path.basename(path))
                sf.write(save_path, audio_seg, SAMPLING_RATE)
            else:
                raise RuntimeError("File: {} Found not > 5s aux".format(path))

if __name__ == "__main__":
    main()


