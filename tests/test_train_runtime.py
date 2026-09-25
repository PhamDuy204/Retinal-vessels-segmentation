from train import parse_args


def test_optimized_our_net_defaults():
    args = parse_args(["--experiment-id", "test-defaults"])

    assert args.model == "our_net"
    assert args.loss == "main_loss"
    assert args.epochs == 60
    assert args.learning_rate == 0.0018
    assert args.num_workers == 4
    assert args.pin_memory is True
    assert args.persistent_workers is True
    assert args.amp is True
    assert args.amp_dtype == "bf16"
    assert args.amp_native_norm is True
    assert args.eval_amp is True
    assert args.eval_amp_dtype == "bf16"
    assert args.eval_start_epoch == 50
    assert args.eval_batch_size == 256
    assert args.micro_batch_size == 48
    assert args.fast_nondeterministic is True
    assert args.fused_adam is True


def test_optimized_defaults_can_be_switched_to_full_fp32():
    args = parse_args([
        "--experiment-id", "test-fp32",
        "--no-amp",
        "--amp-dtype", "fp32",
        "--no-eval-amp",
        "--no-amp-native-norm",
    ])

    assert args.amp is False
    assert args.amp_dtype == "fp32"
    assert args.eval_amp is False
    assert args.amp_native_norm is False
    assert args.fused_adam is True
    assert args.fast_nondeterministic is True


def test_fused_adam_can_be_disabled():
    args = parse_args([
        "--experiment-id", "test-no-fused",
        "--no-fused-adam",
    ])
    assert args.fused_adam is False


def test_stop_after_epoch_is_optional():
    default_args = parse_args(["--experiment-id", "test-stop-default"])
    one_epoch_args = parse_args(
        ["--experiment-id", "test-stop-one", "--stop-after-epoch", "1"]
    )

    assert default_args.stop_after_epoch == 0
    assert one_epoch_args.stop_after_epoch == 1


def test_non_our_net_does_not_auto_enable_native_amp_norm():
    args = parse_args([
        "--experiment-id", "test-unet",
        "--model", "unet",
    ])
    assert args.amp_native_norm is False


def test_main_loss_is_loadable_by_canonical_name():
    from load_model import load_loss_class

    assert load_loss_class("main_loss").__name__ == "MainLoss"
