import argparse
import os
from typing import Optional, Tuple

import numpy as np
import torch
from torch import nn, optim
from torch.nn.utils.clip_grad import clip_grad_norm_
from loguru import logger
from libdf import DF
import random

from df.logger import init_logger, log_metrics, log_model_summary
from df.config import config, Csv
from df.modules import get_device
from df.model import ModelParams
from df.checkpoint import load_model, read_cp, write_cp, check_patience
from df.loss import Loss, Istft
from df.utils import as_real, detach_hidden
from df.lr import cosine_scheduler
from df.dataset_tse_s import make_dataloader

MAX_NANS = 50
log_timings = False
should_stop = False
debug = False
state: Optional[DF] = None

def same_seeds(seed):

    torch.manual_seed(seed)  # Sets the seed for the CPU to generate random numbers
    torch.cuda.manual_seed(seed) # Sets the seed for the multi GPU to generate random numbers
    torch.cuda.manual_seed_all(seed)
    random.seed(seed)
    np.random.seed(seed)
    torch.backends.cudnn.benchmark = False  # Use a fixed convolution algorithm
    torch.backends.cudnn.deterministic = True # Further fixed convolution operation

def main():
    global state, debug

    parser = argparse.ArgumentParser()

    parser.add_argument("--base_dir", default='/Share/wsl/exp/pdf2/exp2_cause/base_dir_demo', type=str)

    parser.add_argument("--log_level", default='INFO', type=str)
    parser.add_argument("--no-debug", action="store_true", dest="debug")
    parser.add_argument("--no-resume", action="store_false", dest="resume")

    args = parser.parse_args()
    os.makedirs(args.base_dir, exist_ok=True)
    checkpoint_dir = os.path.join(args.base_dir, "checkpoints")
    os.makedirs(checkpoint_dir, exist_ok=True)

    debug = args.debug
    log_level = 'DEBUG' if debug else "INFO"
    init_logger(file=os.path.join(args.base_dir, "train.log"), level = log_level, model=args.base_dir)
    config_file = os.path.join(args.base_dir, "config.ini")
    config.load(config_file)
    logger.info("Running on device {}".format(get_device()))

    p = ModelParams()
    device = config("DEVICE", default="", section="train")
    torch.cuda.set_device(device)
    state = DF(
        sr=p.sr, 
        fft_size=p.fft_size, 
        hop_size=p.hop_size, 
        nb_bands=p.nb_erb, 
        min_nb_erb_freqs=p.min_nb_freqs,
        )

    mask_only: bool = config("MASK_ONLY", False, bool, section="train")
    train_df_only: bool = config("DF_ONLY", False, bool, section="train")

    bs:int = config("BATCH_SIZE", 32, int, section="train")
    bs_eval: int = config("BATCH_SIZE_EVAL", 0, int, section="train")
    num_workers = config("NUM_WORKERS", 32, int, section="train")
    log_freq = config("LOG_FREQ", 100, int, section="train")
    seed = config("SEED", 1, int, section="train")

    train_mix_dir = config("TRAIN_MIX_DIR", section="train")
    train_ref_dir = config("train_ref_dir", section="train")
    train_aux_wav_dir = config("train_aux_wav_dir", section="train")


    valid_mix_dir = config("valid_mix_dir", section="train")
    valid_ref_dir = config("valid_ref_dir", section="train")
    valid_aux_wav_dir = config("valid_aux_wav_dir", section="train")


    same_seeds(seed)
    max_sample_len_s = config("max_sample_len_s", 3, int, section="train")
    chunk_size = max_sample_len_s * p.sr

    train_loader = make_dataloader(
                                    True,
                                    train_mix_dir,
                                    train_ref_dir,
                                    train_aux_wav_dir,
                                    batch_size=bs,
                                    num_workers=num_workers,
                                    chunk_size=chunk_size,
                                    df_state=state,
                                    nb_df=p.nb_df,
                                    )
    valid_loader = make_dataloader(
                                    False,
                                    valid_mix_dir,
                                    valid_ref_dir,
                                    valid_aux_wav_dir,
                                    batch_size=bs_eval,
                                    num_workers=num_workers,
                                    chunk_size=chunk_size,
                                    df_state=state,
                                    nb_df=p.nb_df,
                                   )

    max_steps_train = config("max_steps_train", 0, int, section="train")

    lrs = setup_lrs(max_steps_train)
    wds = setup_wds(max_steps_train)

    max_epochs = config("MAX_EPOCHS", 100, int, section="train")
    model, epoch = load_model(
        checkpoint_dir if args.resume else None,
        state,
        jit=False,
        mask_only=mask_only,
        train_df_only=train_df_only,
    )
    assert epoch >= 0
    opt = load_opt(
        checkpoint_dir if args.resume else None,
        model,
        mask_only,
        train_df_only,
    )
    try:
        log_model_summary(model, verbose=args.debug)
    except Exception as e:
        logger.warning(f"Failed to print model summary: {e}")
    
    val_criteria = []
    val_criteria_type = config("VALIDATION_CRITERIA", "loss", section="train")
    val_criteria_rule = config("VALIDATION_CRITERIA_RULE", "min", section="train")
    val_criteria_rule = val_criteria_rule.replace("less", "min").replace("more", "max")
    patience = config("EARLY_STOPPING_PATIENCE", 5, int, section="train")

    losses = setup_losses()
    if config("START_EVAL", False, cast=bool, section="train"):
        val_loss = run_epoch(
                        bs,
                        bs_eval,
                        log_freq,
                        model,
                        opt,
                        losses,
                        split="valid",
                        epoch=epoch-1, 
                        loader=valid_loader,          
        )
        metrics = {"loss": val_loss}
        metrics.update(
            {n: torch.mean(torch.stack(vals)).item() for n, vals in losses.get_summaries()}
        )
        log_metrics(f"[{epoch - 1}] [valid end]", metrics)
    losses.reset_summaries()

    for epoch in range(epoch, max_epochs):
        train_loss = run_epoch(
                        bs,
                        bs_eval,
                        log_freq,
                        model,
                        opt,
                        losses,
                        split="train",
                        epoch=epoch,
                        loader=train_loader,
                        max_steps=max_steps_train,
                        lr_scheduler_values=lrs,
                        wd_scheduler_values=wds,
        )
        metrics = {"loss": train_loss}
        try:
            metrics["lr"] = opt.param_groups[0]["lr"]
        except AttributeError:
            pass
        log_metrics(f"[{epoch}] [train end]", metrics)
        write_cp(opt, "opt", checkpoint_dir, epoch)
        losses.reset_summaries()
        # write_cp(model, "model", checkpoint_dir, epoch)

        val_loss = run_epoch(
                        bs,
                        bs_eval,  
                        log_freq,
                        model,
                        opt,
                        losses,
                        split="valid",
                        epoch=epoch,
                        loader=valid_loader,
        )
        metrics = {"loss": val_loss}
        metrics.update(
            {n: torch.mean(torch.stack(vals)).item() for n, vals in losses.get_summaries()}
        )
        val_criteria = metrics[val_criteria_type]
        log_metrics(f"[{epoch}] [valid end]", metrics)

        write_cp(
            model, "model", checkpoint_dir, epoch, metric=val_criteria, cmp=val_criteria_rule
        )
        
        if not check_patience(
            checkpoint_dir,
            max_patience=patience,
            new_metric=val_criteria,
            cmp=val_criteria_rule,
            raise_=False,
        ):
            break
        losses.reset_summaries()

    logger.info("Finished training and valid")
        

def setup_losses() -> Loss:
    global state, istft
    assert state is not None

    p = ModelParams()

    istft = Istft(p.fft_size, p.hop_size, torch.as_tensor(state.fft_window().copy())).to(
        get_device()
    )
    loss = Loss(state, istft).to(get_device())
    # loss = torch.jit.script(loss)
    return loss

    
def run_epoch(
    bs: int,
    bs_eval: int,
    log_freq: int,
    model: nn.Module,
    opt: optim.Optimizer,
    losses: Loss,
    split: str,
    epoch: int,
    loader: make_dataloader,
    max_steps: Optional[int] = None,
    lr_scheduler_values: Optional[np.ndarray] = None,
    wd_scheduler_values: Optional[np.ndarray] = None,
):

    log_freq = log_freq * 2 if split == "train" else log_freq
    assert split in ("train", "valid")
    batch_size = bs if split == "train" else bs_eval
    logger.info("Start {} epoch {} with batch size {}".format(split, epoch, batch_size))

    dev = get_device()
    loss_mem = []
    is_train = split == "train"
    losses.store_losses = not is_train
    model.train(mode=is_train)
    if max_steps != None:
        start_steps = epoch * max_steps 
    else:
        start_steps = 0
    for i, batch in enumerate(loader):
        opt.zero_grad()
        it = start_steps + i
        # i += 1

        if lr_scheduler_values is not None or wd_scheduler_values is not None:
            for param_group in opt.param_groups:
                if lr_scheduler_values is not None:
                    param_group["lr"] = lr_scheduler_values[it] * param_group.get("lr_scale", 1)
                if wd_scheduler_values is not None and param_group["weight_decay"] > 0:
                    param_group["weight_decay"] = wd_scheduler_values[it]

        feat_spec = batch["feat_spec"].to(dev, non_blocking=True)
        feat_erb = as_real(batch["feat_erb"].to(dev, non_blocking=True))
        noisy = batch["noisy"].to(dev, non_blocking=True)
        clean = batch["clean"].to(dev, non_blocking=True)
        aux_erb = batch['aux_erb'].to(dev, non_blocking=True)
        aux_spec = batch['aux_spec'].to(dev, non_blocking=True)

        with torch.set_grad_enabled(is_train):
            if not is_train:
                input = as_real(noisy).clone()
            else:
                input = as_real(noisy)

            enh, _, _ = model.forward(
                spec=input,
                feat_erb=feat_erb,
                feat_spec=feat_spec,
                aux_erb=aux_erb,
                aux_spec=aux_spec
            )


            # compute loss
            try:
                err = losses.forward(
                    clean,
                    enh,
                )
            except:
                raise

            # BP
            if is_train:
                try:
                    err.backward()
                    clip_grad_norm_(model.parameters(), 1.0, error_if_nonfinite=True)
                except:
                    raise
                opt.step()
            detach_hidden(model)
        loss_mem.append(err.detach())
        if i % log_freq == 0:
            loss_mean = torch.stack(loss_mem[-log_freq:]).mean().cpu()
            loss_dict = {"loss": loss_mean.item()}
            if lr_scheduler_values is not None:
                loss_dict["lr"] = opt.param_groups[0]["lr"]
            if wd_scheduler_values is not None:
                loss_dict["wd"] = opt.param_groups[0]["weight_decay"]
            step = str(i)
            log_metrics(f"[{epoch}/{step}]", loss_dict)

    try:
        cleanup(err, noisy, clean, enh, feat_erb, feat_spec, batch)
    except:
        raise
    return torch.stack(loss_mem).mean().cpu().item()

def cleanup(*args):
    import gc

    for arg in args:
        del arg
    gc.collect()
    torch.cuda.empty_cache()

def load_opt(
    cp_dir: Optional[str],
    model: nn.Module,
    mask_only: bool = False,
    df_only: bool = False,
):
    lr = config("LR", 5e-4, float, section="optim")
    momentum = config("momentum", 0, float, section="optim")  # For sgd, rmsprop
    decay = config("weight_decay", 0.05, float, section="optim")
    optimizer = config("optimizer", "adamw", str, section="optim").lower()
    betas: Tuple[int, int] = config(
        "opt_betas", [0.9, 0.999], Csv(float), section="optim", save=False
    )
    if mask_only:
        params = []
        for n, p in model.named_parameters():
            if not ("dfrnn" in n or "df_dec" in n):
                params.append(p)
    elif df_only:
        params = (p for n, p in model.named_parameters() if "df" in n.lower())
    else:
        params = model.parameters()
    supported = {
        "adam": lambda p: optim.Adam(p, lr=lr, weight_decay=decay, betas=betas, amsgrad=True),
        "adamw": lambda p: optim.AdamW(p, lr=lr, weight_decay=decay, betas=betas, amsgrad=True),
        "sgd": lambda p: optim.SGD(p, lr=lr, momentum=momentum, nesterov=True, weight_decay=decay),
        "rmsprop": lambda p: optim.RMSprop(p, lr=lr, momentum=momentum, weight_decay=decay),
    }
    if optimizer not in supported:
        raise ValueError(
            f"Unsupported optimizer: {optimizer}. Must be one of {list(supported.keys())}"
        )
    opt = supported[optimizer](params)

    logger.debug(f"Training with optimizer {opt}")
    if cp_dir is not None:
        try:
            read_cp(opt, "opt", cp_dir, log=False)
        except ValueError as e:
            logger.error(f"Could not load optimizer state: {e}")
    for group in opt.param_groups:
        group.setdefault("initial_lr", lr)
    return opt

def setup_lrs(steps_per_epoch: int) -> np.ndarray:
    lr = config.get("lr", float, "optim")
    num_epochs = config.get("max_epochs", int, "train")
    lr_min = config("lr_min", 1e-6, float, section="optim")
    lr_warmup = config("lr_warmup", 1e-4, float, section="optim")
    assert lr_warmup < lr
    warmup_epochs = config("warmup_epochs", 3, int, section="optim")
    lr_cycle_mul = config("lr_cycle_mul", 1.0, float, section="optim")
    lr_cycle_decay = config("lr_cycle_decay", 0.5, float, section="optim")
    lr_cycle_epochs = config("lr_cycle_epochs", -1, int, section="optim")
    lr_values = cosine_scheduler(
        lr,
        lr_min,
        epochs=num_epochs,
        niter_per_ep=steps_per_epoch,
        warmup_epochs=warmup_epochs,
        start_warmup_value=lr_warmup,
        initial_ep_per_cycle=lr_cycle_epochs,
        cycle_decay=lr_cycle_decay,
        cycle_mul=lr_cycle_mul,
    )
    return lr_values


def setup_wds(steps_per_epoch: int) -> Optional[np.ndarray]:
    decay = config("weight_decay", 0.05, float, section="optim")
    decay_end = config("weight_decay_end", -1, float, section="optim")
    if decay_end == -1:
        return None
    if decay == 0.0:
        decay = 1e-12
        logger.warning("Got 'weight_decay_end' value > 0, but weight_decay is disabled.")
        logger.warning(f"Setting initial weight decay to {decay}.")
        config.overwrite("optim", "weight_decay", decay)
    num_epochs = config.get("max_epochs", int, "train")
    decay_values = cosine_scheduler(
        decay, decay_end, niter_per_ep=steps_per_epoch, epochs=num_epochs
    )
    return decay_values

if __name__ == "__main__":
    main()





