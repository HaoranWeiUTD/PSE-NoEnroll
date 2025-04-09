from df.config import config
from typing import Tuple, Union
from df.logger import init_logger
from df.model import ModelParams
from df.checkpoint import load_model as load_model_cp
from df.modules import get_device
from df.utils import as_complex, get_norm_alpha, as_complex, as_real
from df.audio_ecapa import WaveReader

import torch
from torch import Tensor, nn
import torchaudio as ta
from torch.utils.data import Dataset
import numpy as np
from libdf import DF, erb, erb_norm, unit_norm
import argparse
import os
from loguru import logger
import time
import warnings


def parse_epoch_type(value: str) -> Union[int, str]:
    try:
        return int(value)
    except ValueError:
        assert value in ("best", "latest")
        return value

def init_df(
    model_base_dir: str = None,
    post_filter: bool = False,
    log_level: str = "INFO",
    log_file: str = "separate_cpu.log",
    config_allow_defaults: bool = False,
    epoch: str = "best",
) -> Tuple[nn.Module, DF, str]:

    log_file = os.path.join(model_base_dir, log_file) if log_file is not None else None
    init_logger(file=log_file, level=log_level, model=model_base_dir)
    config.load(
        os.path.join(model_base_dir, "config.ini"),
        config_must_exist=True,
        allow_defaults=config_allow_defaults,
        allow_reload=True,
    )
    if post_filter:
        config.set("mask_pf", True, bool, ModelParams().section)
        logger.info("Running with post-filter")

    p = ModelParams()
    df_state = DF(
        sr=p.sr,
        fft_size=p.fft_size,
        hop_size=p.hop_size,
        nb_bands=p.nb_erb,
        min_nb_erb_freqs=p.min_nb_freqs,
    )
    checkpoint_dir = os.path.join(model_base_dir, "checkpoints")
    mask_only = config("mask_only", cast=bool, section="train", default=False, save=False)
    model, epoch = load_model_cp(checkpoint_dir, df_state, epoch=epoch, mask_only=mask_only)

    logger.debug(f"Loaded checkpoint from epoch {epoch}")
    model = model.to(get_device())

    logger.info("Running on device {}".format(get_device()))
    logger.info("Model loaded")
    return model, df_state

class My_dataset(Dataset):
    """
    Per Utterance Loader
    """
    def __init__(self, mix_scp, sample_rate):
        self.mix = WaveReader(mix_scp, sample_rate=sample_rate)
        

    
    def __getitem__(self, index):
        key = self.mix.index_keys[index]
        mix = self.mix[key]
        return {
            "mix": mix.astype(np.float32),
		    "key": key
        }
    def __len__(self):
        return len(self.mix)

def main(args, dtype=torch.int16):
    model, df_state = init_df(
        args.model_base_dir,
        post_filter=args.pf,
        log_level=args.log_level,
        config_allow_defaults=True,
        epoch=args.epoch,
    )
    df_sr = ModelParams().sr 
    assert df_sr == 16000

    ds = My_dataset(mix_scp=args.mix_scp, sample_rate=df_sr)

    start = time.time()
    logger.info("start separate audio")
    for i, data_dict in enumerate(ds):
        noisy_td = torch.as_tensor(data_dict["mix"]).unsqueeze(0)
        key = data_dict["key"]
        
        enh = enhance(model, df_state, noisy_td)[0]
        os.makedirs(args.save_path, exist_ok=True)
        file_path = os.path.join(args.save_path, key)
        if enh.dim() < 2:
            enh = enh.unsqueeze(0)
        if dtype == torch.int16 and enh.dtype != torch.int16:
            enh = (enh * (1 << 15)).to(torch.int16)
        ta.save(file_path, enh, df_sr)
    end = time.time()
    logger.info("end separate audio with time of {:.2f}".format(end-start))


def enhance(model: nn.Module, df_state: DF, audio: Tensor):

    model.eval()

    bs = audio.shape[0]
    if hasattr(model, "reset_h0"):
        model.reset_h0(batch_size=bs, device=get_device())

    nb_df = getattr(model, "nb_df", getattr(model, "df_bins", ModelParams().nb_df))
    spec, erb_feat, spec_feat = df_features(audio, df_state, nb_df, device=get_device())
    enhanced = model(spec.clone(), erb_feat, spec_feat)[0].detach().cpu()
    enhanced = as_complex(enhanced.squeeze(1))

    audio = torch.as_tensor(df_state.synthesis(enhanced.numpy()))

    return audio

def df_features(audio: Tensor, df: DF, nb_df: int, device=None):
    spec = df.analysis(audio.numpy())  # [C, Tf] -> [C, Tf, F]
    a = get_norm_alpha(False)
    erb_fb = df.erb_widths()
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", UserWarning)
        erb_feat = torch.as_tensor(erb_norm(erb(spec, erb_fb), a)).unsqueeze(1)
    spec_feat = as_real(torch.as_tensor(unit_norm(spec[..., :nb_df], a)).unsqueeze(1))
    spec = as_real(torch.as_tensor(spec).unsqueeze(1))
    if device is not None:
        spec = spec.to(device)
        erb_feat = erb_feat.to(device)
        spec_feat = spec_feat.to(device)
    return spec, erb_feat, spec_feat

def setup_df_argument_parser(default_log_level: str = "INFO"):
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--model-base-dir",
        "-m",
        type=str,
        default="/node/wsl/exp/exp_pdf2_16k_50h/exp1_down/base_dir",
    )
    parser.add_argument(
        "--pf",
        action="store_true",
    )
    parser.add_argument(
        "--log-level",
        type=str,
        default=default_log_level,
    )
    parser.add_argument(
        "--epoch",
        "-e",
        default="best",
        type=parse_epoch_type,
    )
    return parser

def run():
    parser = setup_df_argument_parser()

    parser.add_argument(
        "--mix_scp",
        type=str,
        default="/Share/wsl/data/DC-data/DNS5_track2_bt_16k/path/noisy.scp",
    )
    parser.add_argument(
        "--save_path",
        type=str,
        default="/node/wsl/exp/exp_pdf2_16k_50h/exp1_down/base_dir/DNSMOS/enhance_wav",
    )
    args = parser.parse_args()
    main(args)
if __name__ == "__main__":
    run()
