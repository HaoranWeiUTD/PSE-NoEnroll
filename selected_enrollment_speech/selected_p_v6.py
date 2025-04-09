# DNSMOS + silero VAD
import librosa
import numpy as np
import onnxruntime as ort
# from rVADfast import rVADfast
import sys
import soundfile as sf
import os
import argparse
from silero_vad import load_silero_vad, get_speech_timestamps

SAMPLING_RATE = 16000
length = 5
CHUNK_SIZE = length * 16000
INPUT_LENGTH = 9.01 

# model = load_silero_vad()

# def vad(wav):
#     speech_timestamps = get_speech_timestamps(wav, model)
#     return speech_timestamps

class ComputeScore:
    def __init__(self, p808_model_path) -> None:
        self.p808_onnx_sess = ort.InferenceSession(p808_model_path)
        self.vad_model = load_silero_vad()
        
    def audio_melspec(self, audio, n_mels=120, frame_size=320, hop_length=160, sr=16000, to_db=True):
        mel_spec = librosa.feature.melspectrogram(y=audio, sr=sr, n_fft=frame_size+1, hop_length=hop_length, n_mels=n_mels)
        if to_db:
            mel_spec = (librosa.power_to_db(mel_spec, ref=np.max)+40)/40
        return mel_spec.T

    def get_polyfit_val(self, sig, bak, ovr, is_personalized_MOS):
        if is_personalized_MOS:
            p_ovr = np.poly1d([-0.00533021,  0.005101  ,  1.18058466, -0.11236046])
            p_sig = np.poly1d([-0.01019296,  0.02751166,  1.19576786, -0.24348726])
            p_bak = np.poly1d([-0.04976499,  0.44276479, -0.1644611 ,  0.96883132])
        else:
            p_ovr = np.poly1d([-0.06766283,  1.11546468,  0.04602535])
            p_sig = np.poly1d([-0.08397278,  1.22083953,  0.0052439 ])
            p_bak = np.poly1d([-0.13166888,  1.60915514, -0.39604546])
        sig_poly = p_sig(sig)
        bak_poly = p_bak(bak)
        ovr_poly = p_ovr(ovr)
        return sig_poly, bak_poly, ovr_poly

    def __call__(self, audio, sampling_rate=16000):
        # self.vad
        speech_timestamps = get_speech_timestamps(audio, self.vad_model)
        total_length = 0
        for timestamps in speech_timestamps:
            length = timestamps['end'] - timestamps['start'] + 1
            total_length += length
        ratio_of_speech = total_length / len(audio)

        fs = sampling_rate
        len_samples = int(INPUT_LENGTH*fs)
        audio_repeat = audio
        while audio_repeat.shape[-1] < len_samples:
            audio_repeat = np.concatenate((audio_repeat, audio), axis=0)
        
        num_hops = int(np.floor(audio_repeat.shape[-1]/fs) - INPUT_LENGTH)+1
        hop_len_samples = fs
        predicted_p808_mos = []
       
        for idx in range(num_hops):
            audio_seg = audio_repeat[int(idx*hop_len_samples):int((idx+INPUT_LENGTH)*hop_len_samples)]
            if audio_seg.shape[-1] < len_samples:
                continue
            p808_input_features = self.audio_melspec(audio=audio_seg[:-160]).astype('float32')[np.newaxis,:,:]
            p808_oi = {'input_1': p808_input_features}
            p808_mos = self.p808_onnx_sess.run(None, p808_oi)[0][0][0]
            predicted_p808_mos.append(p808_mos)
        p808_mos_avg = np.mean(predicted_p808_mos)
        return ratio_of_speech * p808_mos_avg, p808_mos_avg, ratio_of_speech

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--wav_path", default="/Share/wsl/data/librispeech/test-noisy/path/noisy_path.txt")
    parser.add_argument("--save_dir", default="/Share/wsl/data/librispeech/test-noisy/enrol_selecs_DNSMOS_silero")
    parser.add_argument("--log_path", default="/Share/wsl/data/librispeech/test-noisy/log_DNSMOS_silero.txt")
    args = parser.parse_args()
    os.makedirs(args.save_dir, exist_ok=True)
    p808_model_path = "/Share/wsl/exp/QMixCAT/QMixCAT/base_dir/DNSMOS/model_v8.onnx"
    dnsmos_model = ComputeScore(p808_model_path)

    with open(args.log_path, "w") as log, open(args.wav_path, 'r') as f:
        for path in f:
            path = path.strip()
            audio, fs = sf.read(path)
            assert fs == SAMPLING_RATE

            record = {"seg_score":0, "mos_score":0, "speech_ratio":0, "audio_seg":None}
            for start_time in range(0, len(audio), fs):
                audio_seg = audio[start_time:start_time+CHUNK_SIZE]
                if len(audio_seg) < CHUNK_SIZE:
                    continue

                quality_score, mos_score, speech_ratio = dnsmos_model(audio_seg)
                if quality_score > record['seg_score']:
                    record['seg_score'] = quality_score
                    record['mos_score'] = mos_score
                    record['speech_ratio'] = speech_ratio
                    record['audio_seg'] = audio_seg
            save_path = os.path.join(args.save_dir, os.path.basename(path))
            sf.write(save_path, record['audio_seg'], SAMPLING_RATE)
            log.write(f"{path}\t{record['seg_score']}\t{record['mos_score']}\t{record['speech_ratio']}\n")

if __name__ == "__main__":
    main()


