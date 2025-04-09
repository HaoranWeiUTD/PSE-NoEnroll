from collections import defaultdict
from typing import Dict, Final, Iterable, List, Optional, Union

import torch
import torch.nn.functional as F
from torch import Tensor, nn
from libdf import DF

from .config import Csv, config
from .model import ModelParams
from .utils import angle, as_complex, get_device

class Stft(nn.Module):
    def __init__(self, n_fft: int, hop: Optional[int] = None, window: Optional[Tensor] = None, center: bool = True):
        super().__init__()
        self.n_fft = n_fft
        self.hop = hop or n_fft // 4
        self.center = center
        if window is not None:
            assert window.shape[0] == n_fft
        else:
            window = torch.hann_window(self.n_fft)
        self.w: torch.Tensor
        self.register_buffer("w", window)

    def forward(self, input: Tensor):
        # Time-domain input shape: [B, *, T]
        t = input.shape[-1]
        sh = input.shape[:-1]
        out = torch.stft(
            input.reshape(-1, t),
            n_fft=self.n_fft,
            hop_length=self.hop,
            window=self.w,
            center=self.center,
            normalized=True,
            return_complex=True,
        )
        out = out.view(*sh, *out.shape[-2:])
        return out


class Istft(nn.Module):
    def __init__(self, n_fft_inv: int, hop_inv: int, window_inv: Optional[Tensor]=None):
        super().__init__()
        # Synthesis back to time domain
        self.n_fft_inv = n_fft_inv
        self.hop_inv = hop_inv

        assert window_inv.shape[0] == n_fft_inv
        self.w_inv: torch.Tensor
        self.register_buffer("w_inv", window_inv)

    def forward(self, input: Tensor):
        # Input shape: [B, * T, F, (2)]
        input = as_complex(input)
        t, f = input.shape[-2:]
        sh = input.shape[:-2]
        # Even though this is not the DF implementation, it numerical sufficiently close.
        # Pad one extra step at the end to get original signal length
        out = torch.istft(
            F.pad(input.reshape(-1, t, f).transpose(1, 2), (0, 1)),
            n_fft=self.n_fft_inv,
            hop_length=self.hop_inv,
            window=self.w_inv,
            normalized=True,
        )
        if input.ndim > 2:
            out = out.view(*sh, out.shape[-1])
        return out


class SpectralLoss(nn.Module):
    gamma: Final[float]
    f_m: Final[float]
    f_c: Final[float]
    f_u: Final[float]

    def __init__(
        self,
        gamma: float = 1,
        factor_magnitude: float = 1,
        factor_complex: float = 1,
        factor_under: float = 1,
    ):
        super().__init__()
        self.gamma = gamma
        self.f_m = factor_magnitude
        self.f_c = factor_complex
        self.f_u = factor_under

    def forward(self, input, target):
        input = as_complex(input)
        target = as_complex(target)
        input_abs = input.abs()
        target_abs = target.abs()
        if self.gamma != 1:
            input_abs = input_abs.clamp_min(1e-12).pow(self.gamma)
            target_abs = target_abs.clamp_min(1e-12).pow(self.gamma)
        tmp = (input_abs - target_abs).pow(2)
        if self.f_u != 1:
            # Weighting if predicted abs is too low
            tmp *= torch.where(input_abs < target_abs, self.f_u, 1.0)
        loss = torch.mean(tmp) * self.f_m
        if self.f_c > 0:
            if self.gamma != 1:
                input = input_abs * torch.exp(1j * angle.apply(input))
                target = target_abs * torch.exp(1j * angle.apply(target))
            loss_c = (
                F.mse_loss(torch.view_as_real(input), target=torch.view_as_real(target)) * self.f_c
            )
            loss = loss + loss_c
        return loss

class MultiResSpecLoss(nn.Module):
    gamma: Final[float]
    f: Final[float]
    f_complex: Final[Optional[List[float]]]

    def __init__(
        self,
        n_ffts: Iterable[int],
        gamma: float = 1,
        factor: float = 1,
        f_complex: Optional[Union[float, Iterable[float]]] = None,
    ):
        super().__init__()
        self.gamma = gamma
        self.f = factor
        self.stfts = nn.ModuleDict({str(n_fft): Stft(n_fft) for n_fft in n_ffts})
        if f_complex is None or f_complex == 0:
            self.f_complex = None
        elif isinstance(f_complex, Iterable):
            self.f_complex = list(f_complex)
        else:
            self.f_complex = [f_complex] * len(self.stfts)

    def forward(self, input: Tensor, target: Tensor) -> Tensor:
        loss = torch.zeros((), device=input.device, dtype=input.dtype)
        for i, stft in enumerate(self.stfts.values()):
            Y = stft(input)
            S = stft(target)
            Y_abs = Y.abs()
            S_abs = S.abs()
            if self.gamma != 1:
                Y_abs = Y_abs.clamp_min(1e-12).pow(self.gamma)
                S_abs = S_abs.clamp_min(1e-12).pow(self.gamma)
            loss += F.mse_loss(Y_abs, S_abs) * self.f
            if self.f_complex is not None:
                if self.gamma != 1:
                    Y = Y_abs * torch.exp(1j * angle.apply(Y))
                    S = S_abs * torch.exp(1j * angle.apply(S))
                loss += F.mse_loss(torch.view_as_real(Y), torch.view_as_real(S)) * self.f_complex[i]
        return loss

class SiSdr(nn.Module):
    def __init__(self):
        super().__init__()

    def forward(self, input: Tensor, target: Tensor):
        # Input shape: [B, T]
        eps = torch.finfo(input.dtype).eps
        t = input.shape[-1]
        target = target.reshape(-1, t)
        input = input.reshape(-1, t)
        # Einsum for batch vector dot product
        Rss: Tensor = torch.einsum("bi,bi->b", target, target).unsqueeze(-1)
        a: Tensor = torch.einsum("bi,bi->b", target, input).add(eps).unsqueeze(-1) / Rss.add(eps)
        e_true = a * target
        e_res = input - e_true
        Sss = e_true.square()
        Snn = e_res.square()
        # Only reduce over each sample. Supposed to be used when used as a metric.
        Sss = Sss.sum(-1)
        Snn = Snn.sum(-1)
        return 10 * torch.log10(Sss.add(eps) / Snn.add(eps))


class SdrLoss(nn.Module):
    def __init__(self, factor=0.2):
        super().__init__()
        self.factor = factor
        self.sdr = SiSdr()

    def forward(self, input: Tensor, target: Tensor) -> Tensor:
        if self.factor == 0:
            return torch.zeros((), device=input.device)
        # Input shape: [B, T]
        return -self.sdr(input, target).mean() * self.factor
class Loss(nn.Module):

    def __init__(self, state: DF, istft: Optional[Istft] = None):
        
        super().__init__()
        p = ModelParams()
        # self.lsnr = LocalSnrTarget(ws=20, target_snr_range=[p.lsnr_min - 5, p.lsnr_max + 5])
        self.istft = istft  # Could also be used for sdr loss
        self.sr = p.sr
        self.fft_size = p.fft_size
        self.nb_df = p.nb_df
        self.store_losses = False
        self.summaries: Dict[str, List[Tensor]] = self.reset_summaries()
       

        # SpectralLoss
        self.sl_fm = config("factor_magnitude", 0, float, section="SpectralLoss")  # e.g. 1e4
        self.sl_fc = config("factor_complex", 0, float, section="SpectralLoss")
        self.sl_fu = config("factor_under", 1, float, section="SpectralLoss")
        self.sl_gamma = config("gamma", 1, float, section="SpectralLoss")
        self.sl_f = self.sl_fm + self.sl_fc
        if self.sl_f > 0:
            self.sl = SpectralLoss(
                factor_magnitude=self.sl_fm,
                factor_complex=self.sl_fc,
                factor_under=self.sl_fu,
                gamma=self.sl_gamma,
            )
        else:
            self.sl = None

        # Multi Resolution Spectrogram Loss
        self.mrsl_f = config("factor", 0, float, section="MultiResSpecLoss")
        self.mrsl_fc = config("factor_complex", 0, float, section="MultiResSpecLoss")
        self.mrsl_gamma = config("gamma", 1, float, section="MultiResSpecLoss")
        self.mrsl_ffts: List[int] = config("fft_sizes", [512, 1024, 2048], Csv(int), section="MultiResSpecLoss")  # type: ignore
        if self.mrsl_f > 0:
            assert istft is not None
            self.mrsl = MultiResSpecLoss(self.mrsl_ffts, self.mrsl_gamma, self.mrsl_f, self.mrsl_fc)
        else:
            self.mrsl = None
        
        # Local snr Loss
        
        # SiSnr Loss
        self.sisnr_f = config("factor", 1, float, section="SISNR")
        if self.sisnr_f > 0:
            self.sisnr = SdrLoss(self.sisnr_f)
        else:
            self.sisnr = None

        self.dev_str = get_device().type

    def forward(
        self,
        clean: Tensor,
        enhanced: Tensor,
    ):

        enhanced_td = None
        clean_td = None

        if self.istft is not None:
            enhanced_td = self.istft(enhanced)
            clean_td = self.istft(clean)

        sl, mrsl, sisnr = [torch.zeros((), device=clean.device)] * 3
        # compute spectralloss
        if self.sl_f != 0 and self.sl is not None:
            sl = self.sl(input=enhanced, target=clean)

        # compute multresspecralloss
        if self.mrsl_f > 0 and self.mrsl is not None:
            mrsl = self.mrsl(enhanced_td, clean_td)
    
        # compute sisnr
        if self.sisnr_f > 0 and self.sisnr is not None:
            sisnr = self.sisnr(input=enhanced_td, target=clean_td)

        if self.store_losses:
            assert clean_td is not None
            self.store_summaries(
                sl,
                mrsl,
                sisnr,
            )

        return sl + mrsl + sisnr

    def reset_summaries(self):
        self.summaries = defaultdict(list)
        return self.summaries

    def get_summaries(self):
        return self.summaries.items()

    def store_summaries(
        self,
        sl: Tensor,
        mrsl: Tensor,
        sisnr: Tensor,
    ):
        self.summaries["SpectralLoss"].append(sl.detach())
        self.summaries["MultiResSpecLoss"].append(mrsl.detach())
        self.summaries["SiSnrLoss"].append(sisnr.detach())



