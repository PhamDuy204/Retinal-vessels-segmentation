from train import parse_args


def test_fused_adam_is_opt_in():
    default_args = parse_args(["--experiment-id", "test-default"])
    fused_args = parse_args(["--experiment-id", "test-fused", "--fused-adam"])

    assert default_args.fused_adam is False
    assert fused_args.fused_adam is True


def test_stop_after_epoch_is_optional():
    default_args = parse_args(["--experiment-id", "test-stop-default"])
    one_epoch_args = parse_args(
        ["--experiment-id", "test-stop-one", "--stop-after-epoch", "1"]
    )

    assert default_args.stop_after_epoch == 0
    assert one_epoch_args.stop_after_epoch == 1
