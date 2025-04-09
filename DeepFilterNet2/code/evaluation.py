
from df.logger import init_logger
import os
from loguru import logger
import time
import numpy as np
import argparse
import soundfile as sf

from pesq import pesq
from pystoi.stoi import stoi
from df.sepm import composite as composite_py

SAMPLE_RATE = 16000

def main(args):
    log_level=args.log_level
    log_file=args.log_file
    # model_base_dir = args.model_base_dir
    init_logger(file=log_file, level=log_level)

    refence_dir = args.refence_dir
    noisy_dir = args.noisy_dir
    enhance_dir = args.enhance_dir
    
    total_cnt = 0
    total_PESQ = total_STOI = total_SISNR = total_CSIG = total_CBAK = total_COVL = 0
    total_PESQi = total_STOIi = total_SISNRi = total_CSIGi = total_CBAKi = total_COVLi = 0

    start = time.time()
    for filename in sorted(os.listdir(refence_dir)):
        if filename.endswith('.wav'):
            ref_audio, fr = sf.read(os.path.join(refence_dir, filename), dtype='float32')
            noisy_audio, fn = sf.read(os.path.join(noisy_dir, filename), dtype='float32')
            enh_audio, fe = sf.read(os.path.join(enhance_dir, filename), dtype='float32')
            assert fr == fn == fe == SAMPLE_RATE
            if enh_audio.size != noisy_audio.size:
                end = min(enh_audio.size, noisy_audio.size)
                enh_audio = enh_audio[:end]
                noisy_audio = noisy_audio[:end]
                ref_audio = ref_audio[:end]

            PESQ, PESQi = cal_PESQi(enh_audio, ref_audio, noisy_audio)
            STOI, STOIi = cal_STOIi(enh_audio, ref_audio, noisy_audio)
            SISNR, delta = cal_SISDRi(enh_audio, ref_audio, noisy_audio)

            CSIG, CBAK, COVL, CSIGi, CBAKi, COVLi= cal_compositei(enh_audio, ref_audio, noisy_audio)
            
            logger.info("Utt={:d} | PESQ={:.3f} | PESQi={:.3f} | STOI={:.3f} | STOIi={:.3f} | "
                        "SI-SNR={:.3f} | SI-SNRi={:.3f} | CSIG={:.3f} | CSIGi={:.3f} | CBAK={:.3f} | "
                        "CBAKi={:.3f} | COVL={:.3f} | COVLi={:.3f}" .format(total_cnt+1, PESQ, \
                            PESQi, STOI, STOIi, SISNR, delta, CSIG, CSIGi, CBAK, CBAKi, COVL, COVLi))
        
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
    end = time.time()
    logger.info("Average STOI: {:.2f}".format(total_STOI / total_cnt))
    logger.info("Average STOIi: {:.2f}".format(total_STOIi / total_cnt))
    logger.info("Average SI-SNR: {:.3f}".format(total_SISNR / total_cnt))
    logger.info("Average SI-SNRi: {:.3f}".format(total_SISNRi / total_cnt))
    
    logger.info('Time Elapsed: {:.1f}s'.format(end-start))



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
        "--log-level",
        type=str,
        default=default_log_level,
    )
    return parser

def run():
    parser = setup_df_argument_parser()
    parser.add_argument(
        "--noisy_dir",
        type=str,
        default="/Share/wsl/data/librispeech/test-noisy/noisy",
    )
    parser.add_argument(
        "--refence_dir",
        type=str,
        default="/Share/wsl/data/librispeech/test-noisy/clean",
    )
    parser.add_argument(
        "--enhance_dir",
        type=str,
        default="/Share/wsl/exp/expv100/data3/exp/exp_paper/exp_9/enhance_librimixed_testset-5s-2620",
    )
    parser.add_argument(
        "--log_file",
        type=str,
        default='/Share/wsl/exp/expv100/data3/exp/exp_paper/exp_9/enhance_librimixed_testset-5s-2620.log',
    )
    args = parser.parse_args()
    main(args)
if __name__ == "__main__":
    run()
