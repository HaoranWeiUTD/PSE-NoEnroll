# 方法2：直接使用sil-VAD
import os 
import soundfile as sf
# from rVADfast import rVADfast
import numpy as np
import argparse
import librosa
import numpy as np
from silero_vad import load_silero_vad, get_speech_timestamps

model = load_silero_vad()

def vad(wav):
    speech_timestamps = get_speech_timestamps(wav, model)
    
    return speech_timestamps

def find_non_speech_segments(speech_segments, total_length):
    non_speech_segments = []
    
    # 如果没有语音片段，则整个音频都是非语音部分
    if not speech_segments:
        return [{'start': 0, 'end': total_length - 1}]
    
    # 检查第一个语音片段前的非语音部分
    if speech_segments[0]['start'] > 0:
        non_speech_segments.append({'start': 0, 'end': speech_segments[0]['start'] - 1})
    
    # 检查语音片段之间的非语音部分
    for i in range(1, len(speech_segments)):
        start = speech_segments[i - 1]['end'] + 1
        end = speech_segments[i]['start'] - 1
        if start <= end:
            non_speech_segments.append({'start': start, 'end': end})
    
    # 检查最后一个语音片段后的非语音部分
    if speech_segments[-1]['end'] < total_length - 1:
        non_speech_segments.append({'start': speech_segments[-1]['end'] + 1, 'end': total_length - 1})
    
    return non_speech_segments

def add_fade_in_out(audio_data, sample_rate, fade_in_duration=0.05, fade_out_duration=0.05):
    # 计算淡入和淡出的样本数量
    fade_in_samples = int(sample_rate * fade_in_duration)
    fade_out_samples = int(sample_rate * fade_out_duration)
    
    # 创建淡入和淡出的增益系数
    fade_in = np.linspace(0, 1, fade_in_samples)
    fade_out = np.linspace(1, 0, fade_out_samples)
    
    # 应用淡入效果
    audio_data[:fade_in_samples] *= fade_in
    
    # 应用淡出效果
    audio_data[-fade_out_samples:] *= fade_out
    
    return audio_data

def main(args):
    noisy_dir = args.noisy_dir
    save_path = args.save_path
    for filename in sorted(os.listdir(noisy_dir)):
        if filename.endswith('.wav'):
            noisy_audio, fs = sf.read(os.path.join(noisy_dir, filename), dtype='float32')

            speech_segments = vad(noisy_audio)
            non_speech_segments = find_non_speech_segments(speech_segments, len(noisy_audio))
            noise_all = []
            for non_speech_seg in non_speech_segments:
                length = non_speech_seg['end'] - non_speech_seg['start']
                if length < 1 * fs:
                    continue
                non_speech = noisy_audio[non_speech_seg['start'] : non_speech_seg['end']]
                # 使用librosa检测非静音段, --top_db 表示能量比最大幅值能量小15db一下的部分看做静音段
                noise_seg_idx = librosa.effects.split(non_speech, top_db=15)
                noises = []
                for idx in noise_seg_idx:
                    noises.append(non_speech[idx[0] : idx[1]])
                noise = np.concatenate(noises)

                noise = add_fade_in_out(noise, fs)
                noise_all.append(noise)

            cur_noise = np.concatenate(noise_all)
            os.makedirs(save_path, exist_ok=True)

            sf.write(os.path.join(save_path, filename), cur_noise, fs)
            print(len(cur_noise))
    print('done!')

if __name__ =="__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--noisy_dir",
        type=str,
        default="/Share/wsl/data/win/youtube_data/wav",
    )
    parser.add_argument(
        "--save_path",
        type=str,
        default="/Share/wsl/data/win/youtube_data/ests_noise_v1",
    )
    args = parser.parse_args()
    main(args)