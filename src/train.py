import numpy as np
import torch
import torch.nn as nn
import timm

from .dataset import (
    AugmentWrapper,
    SpectrogramDataset,
)


@torch.no_grad()
def compute_stats_from_paths(
    train_paths,
):
    """Compute channel-wise mean/std from training samples only."""
    if not train_paths:
        return {
            "mean": [0.5, 0.5, 0.5],
            "std": [0.5, 0.5, 0.5],
        }

    first_arr = np.load(
        train_paths[0]
    )
    channels = (
        first_arr.shape[0]
        if first_arr.ndim == 3
        else 1
    )

    ch_sum = torch.zeros(
        channels,
        dtype=torch.float64,
    )
    ch_sqsum = torch.zeros(
        channels,
        dtype=torch.float64,
    )
    n_px = 0

    for path in train_paths:
        arr = np.load(path)
        arr = np.nan_to_num(
            arr,
            nan=0.0,
            posinf=0.0,
            neginf=0.0,
        )

        tensor = torch.from_numpy(
            arr
        )
        flat = tensor.reshape(
            channels,
            -1,
        ).double()

        ch_sum += flat.sum(
            dim=1
        )
        ch_sqsum += (
            flat ** 2
        ).sum(
            dim=1
        )

        n_px += (
            tensor.shape[1]
            * tensor.shape[2]
        )

    mean = (
        ch_sum
        / max(1, n_px)
    ).float()

    var = (
        ch_sqsum
        / max(1, n_px)
    ).float() - mean ** 2

    std = torch.sqrt(
        torch.clamp(
            var,
            min=1e-8,
        )
    )

    return {
        "mean": mean.tolist(),
        "std": std.tolist(),
    }


def make_subset_from_indices(
    root,
    base_stats,
    indices,
    augment=False,
    noise_std=0.0,
    strong_specaugment=False,
):
    """Create a dataset subset using shared training statistics."""
    ds = SpectrogramDataset(
        root,
        stats=base_stats,
        augment=augment,
        index_json=None,
    )

    ds.image_paths = [
        ds.image_paths[i]
        for i in indices
    ]
    ds.labels = ds.labels[
        indices
    ]

    if hasattr(
        ds,
        "groups",
    ):
        ds.groups = [
            ds.groups[i]
            for i in indices
        ]

    if augment and (
        noise_std > 0.0
        or strong_specaugment
    ):
        ds = AugmentWrapper(
            ds,
            noise_std=noise_std,
            strong_specaugment=strong_specaugment,
        )

    return ds


def build_model(
    num_classes: int,
    in_chans: int = 3,
    head_dropout: float = 0.3,
):
    """Build ConvNeXtV2-Tiny classifier."""
    model_name = (
        "convnextv2_tiny.fcmae"
    )

    print(
        f"Using model: "
        f"{model_name}"
    )

    model = timm.create_model(
        model_name,
        pretrained=True,
        num_classes=0,
        in_chans=in_chans,
    )

    in_features = (
        model.num_features
    )

    model.head = nn.Sequential(
        nn.AdaptiveAvgPool2d(1),
        nn.Flatten(1),
        nn.LayerNorm(
            in_features
        ),
        nn.Dropout(
            p=head_dropout
        ),
        nn.Linear(
            in_features,
            num_classes,
        ),
    )

    return model


def mixup_batch(
    x: torch.Tensor,
    y: torch.Tensor,
    alpha: float = 0.4,
):
    if alpha <= 0:
        return (
            x,
            y,
            y,
            1.0,
        )

    lam = np.random.beta(
        alpha,
        alpha,
    )

    idx = torch.randperm(
        x.size(0),
        device=x.device,
    )

    x_mix = (
        lam * x
        + (1 - lam) * x[idx]
    )

    return (
        x_mix,
        y,
        y[idx],
        float(lam),
    )


def mixup_loss(
    criterion,
    logits,
    y_a,
    y_b,
    lam: float,
):
    return (
        lam * criterion(
            logits,
            y_a,
        )
        + (1 - lam)
        * criterion(
            logits,
            y_b,
        )
    )


@torch.no_grad()
def mixup_expected_acc(
    pred: torch.Tensor,
    y_a: torch.Tensor,
    y_b: torch.Tensor,
    lam: float,
):
    pa = (
        pred == y_a
    ).float()

    pb = (
        pred == y_b
    ).float()

    return float(
        (
            lam * pa
            + (1 - lam) * pb
        ).mean().item()
    )


def build_optimizer_and_scheduler(
    model,
    base_lr,
    epochs,
    freeze_epochs=5,
    weight_decay=1e-4,
):
    """Train the head first, then unfreeze the backbone."""
    for param in model.parameters():
        param.requires_grad = False

    for param in model.head.parameters():
        param.requires_grad = True

    optimizer = torch.optim.AdamW(
        filter(
            lambda p: p.requires_grad,
            model.parameters(),
        ),
        lr=base_lr,
        weight_decay=weight_decay,
    )

    warmup_epochs = max(
        1,
        min(
            5,
            freeze_epochs,
        ),
    )

    main_epochs = max(
        1,
        epochs - warmup_epochs,
    )

    warmup = (
        torch.optim.lr_scheduler.LinearLR(
            optimizer,
            start_factor=0.2,
            end_factor=1.0,
            total_iters=warmup_epochs,
        )
    )

    cosine = (
        torch.optim.lr_scheduler.CosineAnnealingLR(
            optimizer,
            T_max=main_epochs,
        )
    )

    scheduler = (
        torch.optim.lr_scheduler.SequentialLR(
            optimizer,
            [
                warmup,
                cosine,
            ],
            milestones=[
                warmup_epochs
            ],
        )
    )

    return (
        optimizer,
        scheduler,
        warmup_epochs,
    )


def unfreeze_backbone_and_reset_opt(
    model,
    current_epoch,
    epochs,
    base_lr,
    weight_decay=1e-4,
):
    for param in model.parameters():
        param.requires_grad = True

    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=base_lr,
        weight_decay=weight_decay,
    )

    remain = max(
        1,
        epochs
        - current_epoch
        - 1,
    )

    warmup = (
        torch.optim.lr_scheduler.LinearLR(
            optimizer,
            start_factor=0.2,
            end_factor=1.0,
            total_iters=1,
        )
    )

    cosine = (
        torch.optim.lr_scheduler.CosineAnnealingLR(
            optimizer,
            T_max=remain,
        )
    )

    scheduler = (
        torch.optim.lr_scheduler.SequentialLR(
            optimizer,
            [
                warmup,
                cosine,
            ],
            milestones=[1],
        )
    )

    return (
        optimizer,
        scheduler,
    )


def epoch_loop(
    model,
    loader,
    criterion,
    device,
    train_mode=True,
    optimizer=None,
    max_grad=5.0,
    use_mixup=True,
    mixup_alpha=0.4,
    scaler=None,
    use_amp=True,
):
    """Run one training or validation epoch."""
    model.train(
        mode=train_mode
    )

    total_loss = 0.0
    acc_sum = 0.0
    total_seen = 0

    amp_enabled = bool(
        use_amp
        and device.type == "cuda"
    )

    for x, y in loader:
        x = torch.nan_to_num(
            x.to(
                device,
                non_blocking=True,
            ),
            nan=0.0,
            posinf=0.0,
            neginf=0.0,
        )

        y = y.to(
            device,
            non_blocking=True,
        )

        if train_mode:
            if optimizer is None:
                raise ValueError(
                    "optimizer is required "
                    "when train_mode=True"
                )

            optimizer.zero_grad(
                set_to_none=True
            )

        with torch.set_grad_enabled(
            train_mode
        ):
            with torch.amp.autocast(
                device_type="cuda",
                dtype=torch.float16,
                enabled=amp_enabled,
            ):
                if (
                    train_mode
                    and use_mixup
                    and mixup_alpha > 0
                ):
                    (
                        x_mix,
                        y_a,
                        y_b,
                        lam,
                    ) = mixup_batch(
                        x,
                        y,
                        alpha=mixup_alpha,
                    )

                    logits = model(
                        x_mix
                    )

                    loss = mixup_loss(
                        criterion,
                        logits,
                        y_a,
                        y_b,
                        lam,
                    )

                    pred = logits.argmax(
                        dim=1
                    )

                    batch_acc = (
                        mixup_expected_acc(
                            pred,
                            y_a,
                            y_b,
                            lam,
                        )
                    )

                else:
                    logits = model(x)
                    loss = criterion(
                        logits,
                        y,
                    )

                    pred = logits.argmax(
                        dim=1
                    )

                    batch_acc = float(
                        (
                            pred == y
                        ).float().mean().item()
                    )

            if train_mode:
                if (
                    scaler is not None
                    and scaler.is_enabled()
                ):
                    scaler.scale(
                        loss
                    ).backward()

                    scaler.unscale_(
                        optimizer
                    )
                else:
                    loss.backward()

                for param in model.parameters():
                    if param.grad is not None:
                        torch.nan_to_num_(
                            param.grad,
                            nan=0.0,
                            posinf=1.0,
                            neginf=-1.0,
                        )
                        param.grad.clamp_(
                            -max_grad,
                            max_grad,
                        )

                if (
                    scaler is not None
                    and scaler.is_enabled()
                ):
                    scaler.step(
                        optimizer
                    )
                    scaler.update()
                else:
                    optimizer.step()

        batch_size = y.size(0)

        total_loss += (
            float(
                loss.item()
            )
            * batch_size
        )

        acc_sum += (
            batch_acc
            * batch_size
        )

        total_seen += (
            batch_size
        )

    return (
        total_loss
        / max(1, total_seen),
        acc_sum
        / max(1, total_seen),
    )
