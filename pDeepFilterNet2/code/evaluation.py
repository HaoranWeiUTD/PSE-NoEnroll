from df.dataset_tse_s import My_dataset
from df.logger import init_logger
from df.config import config
from df.model import ModelParams
from df.checkpoint import load_model as load_model_cp
from df.modules import get_device
from df.utils import as_complex, get_norm_alpha, as_complex, as_real

from libdf import DF, erb, erb_norm, unit_norm
import torch
import warnings
from torch import Tensor, nn
from typing import Tuple, Union
import os
from loguru import logger
import time
import numpy as np
import argparse
# import librosa

from pesq import pesq
from pystoi.stoi import stoi
from df.sepm import composite as composite_py

SAMPLE_RATE = 16000

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
    log_file: str = "enhance_down.log",
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

    suffix = os.path.basename(os.path.abspath(model_base_dir))
    if post_filter:
        suffix += "_pf"
    logger.info("Running on device {}".format(get_device()))
    logger.info("Model loaded")
    return model, df_state, suffix

def main(args):

    model, df_state, _ = init_df(
        args.model_base_dir,
        post_filter=args.pf,
        log_level=args.log_level,
        log_file=args.log_file,
        config_allow_defaults=True,
        epoch=args.epoch,
    )
    total_cnt = 0
    total_PESQ = total_STOI = total_SISNR = total_CSIG = total_CBAK = total_COVL = 0
    total_PESQi = total_STOIi = total_SISNRi = total_CSIGi = total_CBAKi = total_COVLi = 0

    df_sr = ModelParams().sr
    assert df_sr == SAMPLE_RATE
    
    ds = My_dataset(args.mix_scp, args.ref_scp, args.aux_scp, sample_rate=df_sr)

    # n_samples = len(ds)
    start = time.time()
    for i, data_dict in enumerate(ds):
        noisy_td = torch.as_tensor(data_dict["mix"]).unsqueeze(0)
        clean_td = torch.as_tensor(data_dict["ref"]).unsqueeze(0)
        aux_td = torch.as_tensor(data_dict["aux_wav"]).unsqueeze(0)

        ests_16 = enhance(model, df_state, noisy_td, aux_td)[0]
        ref_16 = df_state.synthesis(df_state.analysis(clean_td.numpy()))[0]
        mix_16 = df_state.synthesis(df_state.analysis(noisy_td.numpy()))[0]
        
        PESQ, PESQi = cal_PESQi(ests_16, ref_16, mix_16)
        STOI, STOIi = cal_STOIi(ests_16, ref_16, mix_16)
        SISNR, delta = cal_SISDRi(ests_16, ref_16, mix_16)

        CSIG, CBAK, COVL, CSIGi, CBAKi, COVLi= cal_compositei(ests_16, ref_16, mix_16)
        
        logger.info("Utt={:d} | PESQ={:.3f} | PESQi={:.3f} | STOI={:.3f} | STOIi={:.3f} | SI-SNR={:.3f} | SI-SNRi={:.3f} | CSIG={:.3f} | CSIGi={:.3f} | CBAK={:.3f} | CBAKi={:.3f} | COVL={:.3f} | COVLi={:.3f}" \
                .format(total_cnt+1, PESQ, PESQi, STOI, STOIi, SISNR, delta, CSIG, CSIGi, CBAK, CBAKi, COVL, COVLi))
        
        
        total_PESQ += PESQ
        total_PESQi += PESQi
        total_STOI += STOI
        total_STOIi += STOIi
        total_SISNR += SISNR
        total_SISNRi += delta

        total_CSIG += CSIG
        total_CSIGi += CSIGi
        total_CBAK += CBAK
        total_CBAKi += CBAKi
        total_COVL += COVL
        total_COVLi += COVLi
        total_cnt += 1  

    end = time.time()
    logger.info("Average PESQ: {:.3f}".format(total_PESQ / total_cnt))
    logger.info("Average PESQi: {:.3f}".format(total_PESQi / total_cnt))
    logger.info("Average STOI: {:.2f}".format(total_STOI / total_cnt))
    logger.info("Average STOIi: {:.2f}".format(total_STOIi / total_cnt))
    logger.info("Average SI-SNR: {:.3f}".format(total_SISNR / total_cnt))
    logger.info("Average SI-SNRi: {:.3f}".format(total_SISNRi / total_cnt))

    logger.info("Average CSIG: {:.2f}".format(total_CSIG / total_cnt))
    logger.info("Average CSIGi: {:.2f}".format(total_CSIGi / total_cnt))
    logger.info("Average CBAK: {:.2f}".format(total_CBAK / total_cnt))
    logger.info("Average CBAKi: {:.2f}".format(total_CBAKi / total_cnt))
    logger.info("Average COVL: {:.2f}".format(total_COVL / total_cnt))
    logger.info("Average COVLi: {:.2f}".format(total_COVLi / total_cnt))
    
    logger.info('Time Elapsed: {:.1f}s'.format(end-start))


def enhance(model: nn.Module, df_state: DF, audio: Tensor, aux_td: Tensor):
    model.eval()
    bs = audio.shape[0]
    if hasattr(model, "reset_h0"):
        model.reset_h0(batch_size=bs, device=get_device())

    nb_df = getattr(model, "nb_df", getattr(model, "df_bins", ModelParams().nb_df))
    spec, erb_feat, spec_feat = df_features(audio, df_state, nb_df, device=get_device())
    _, aux_erb_feat, aux_spec_feat = df_features(aux_td, df_state, nb_df, device=get_device())
    
    enhanced = model(spec.clone(), erb_feat, spec_feat, aux_erb_feat, aux_spec_feat)[0].detach().cpu()
    enhanced = as_complex(enhanced.squeeze(1))
    audio = df_state.synthesis(enhanced.numpy())

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

def cal_PESQ(est, ref):
    assert len(est) == len(ref)
    mode ='wb'
    p = pesq(SAMPLE_RATE, ref, est, mode)
    return p

def cal_PESQi(est, ref, mix):
    assert len(est) == len(ref) == len(mix)
    pesq1 = cal_PESQ(est, ref)
    pesq2 = cal_PESQ(mix, ref)

    return pesq1, pesq1 - pesq2

def cal_STOI(est, ref):
    assert len(est) == len(ref)
    p = stoi(ref, est, SAMPLE_RATE)
    return p * 100


def cal_STOIi(est, ref, mix):
    """Calcuate Scale-Invariant Source-to-Noise Ratio (SI-SNR)
    Args:
        est: separated signal, numpy.ndarray, [T]
        ref: reference signal, numpy.ndarray, [T]
    Returns:
        SISNR
    """
    assert len(est) == len(ref) == len(mix)
    stoi1 = cal_STOI(est, ref)
    stoi2 = cal_STOI(mix, ref)

    return stoi1, stoi1 - stoi2

def cal_SISDR(estimate: np.ndarray, reference: np.ndarray):
    reference = reference.reshape(-1, 1)
    estimate = estimate.reshape(-1, 1)
    eps = np.finfo(reference.dtype).eps
    Rss = np.dot(reference.T, reference)

    # get the scaling factor for clean sources
    a = (eps + np.dot(reference.T, estimate)) / (Rss + eps)
    e_true = a * reference
    e_res = estimate - e_true
    Sss = (e_true**2).sum()
    Snn = (e_res**2).sum()
    sisdr = 10 * np.log10((eps + Sss) / (eps + Snn))
    return sisdr

def cal_SISDRi(est, ref, mix):
    """Calcuate Scale-Invariant Source-to-Noise Ratio (SI-SNR)
    Args:
        est: separated signal, numpy.ndarray, [T]
        ref: reference signal, numpy.ndarray, [T]
    Returns:
        SISNR
    """
    assert len(est) == len(ref) == len(mix)
    stoi1 = cal_SISDR(est, ref)
    stoi2 = cal_SISDR(mix, ref)

    return stoi1, stoi1 - stoi2

def cal_composite(est, ref, sr=SAMPLE_RATE):

    assert len(est) == len(ref)
    pesq_mos, Csig, Cbak, Covl, segSNR = composite_py(ref, est, sr)

    return pesq_mos, Csig, Cbak, Covl, segSNR

def cal_compositei(est, ref, mix):
    assert len(est) == len(ref) == len(mix)
    pesq_mos1, Csig1, Cbak1, Covl1, segSNR1 = cal_composite(est, ref)
    pesq_mos2, Csig2, Cbak2, Covl2, segSNR2 = cal_composite(mix, ref)

    return Csig1, Cbak1, Covl1, Csig1-Csig2, Cbak1-Cbak2, Covl1-Covl2

def setup_df_argument_parser(default_log_level: str = "INFO"):
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--model-base-dir",
        "-m",
        type=str,
        default="/Share/wsl/exp/expv100/data3/exp/exp_paper/exp_9/base_dir",
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
        default="/Share/wsl/data/librispeech/test-noisy/path_v1/noisy.scp",
    )
    parser.add_argument(
        "--ref_scp",
        type=str,
        default="/Share/wsl/data/librispeech/test-noisy/path_v1/clean.scp",
    )
    parser.add_argument(
        "--aux_scp",
        type=str,
        default="/Share/wsl/data/librispeech/test-noisy/path_v1/aux-5s.scp",
    )
    parser.add_argument(
        "--log_file",
        type=str,
        default='enh_librimixed_2620_gpu.log',
    )
    args = parser.parse_args()
    main(args)
if __name__ == "__main__":
    run()
