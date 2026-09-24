from run_statistics import common_args_to_cli


def test_false_values_for_default_true_flags_emit_negative_switches():
    cli = common_args_to_cli(
        {
            "amp": False,
            "eval_amp": False,
            "amp_native_norm": False,
            "fused_adam": False,
            "fast_nondeterministic": False,
            "pin_memory": False,
            "persistent_workers": False,
        }
    )

    assert "--no-amp" in cli
    assert "--no-eval-amp" in cli
    assert "--no-amp-native-norm" in cli
    assert "--no-fused-adam" in cli
    assert "--no-fast-nondeterministic" in cli
    assert "--no-pin-memory" in cli
    assert "--no-persistent-workers" in cli


def test_false_values_for_store_true_flags_remain_omitted():
    cli = common_args_to_cli(
        {
            "tf32": False,
            "compile_model": False,
            "channels_last": False,
        }
    )

    assert cli == []
