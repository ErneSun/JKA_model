from __future__ import annotations

from pathlib import Path

from eval.evaluate_v0_7 import evaluate_v0_7
from jka_model.config import load_config
from jka_model.data import ChannelStandardizer, data_fingerprint, make_split_manifest
from jka_model.problems import create_problem_adapter
from jka_model.training import TrainStage
from jka_model.utils import Checkpoint, save_checkpoint
from train.prepare_v0_7 import prepare_v0_7_cache
from train.train_v0_6 import initialize_v0_6_model
from train.train_v0_7 import train_v0_7


def _v0_6_checkpoint(tmp_path: Path) -> Path:
    config = load_config("configs/v0_6/advection_diffusion_2d_cpu_smoke.yaml")
    adapter = create_problem_adapter(config)
    records = adapter.build_dataset(seed=config.training.seed)
    spec = adapter.build_problem_spec()
    manifest = make_split_manifest(records, config.data.split)
    normalizer = ChannelStandardizer(eps=config.data.normalization.eps).fit(records, manifest, spec)
    model = initialize_v0_6_model(config, device="cpu")
    path = tmp_path / "v06.pt"
    save_checkpoint(
        Checkpoint(
            train_stage=TrainStage.JEPA,
            epoch=1,
            global_step=1,
            optimizer_update_step=1,
            online_model_state=model.online_state_dict(),
            target_model_state=model.target_encoder.state_dict(),
            normalizer_state=normalizer.state_dict(),
            problem_spec=spec,
            config=config,
            data_fingerprint=data_fingerprint(records, spec),
            split_manifest=manifest.to_dict(),
        ),
        path,
    )
    return path


def test_cpu_cache_train_and_standalone_checkpoint_smoke(tmp_path: Path) -> None:
    config = load_config("configs/v0_7/advection_diffusion_2d_cpu_smoke.yaml")
    source = _v0_6_checkpoint(tmp_path)
    cache_path = tmp_path / "cache.pt"
    diagnostics_path = tmp_path / "diagnostics.json"
    prepare_v0_7_cache(
        config,
        backbone_checkpoint=source,
        destination=cache_path,
        diagnostics_path=diagnostics_path,
        device="cpu",
    )
    zero = train_v0_7(
        config,
        backbone_checkpoint=source,
        cache_path=cache_path,
        variant="zero",
        run_dir=tmp_path / "zero",
        device="cpu",
    )
    history = train_v0_7(
        config,
        backbone_checkpoint=source,
        cache_path=cache_path,
        variant="history",
        run_dir=tmp_path / "history",
        device="cpu",
    )
    assert zero.best_checkpoint.is_file()
    assert history.best_checkpoint.is_file()
    assert history.completed_epochs >= 1
    evaluation = evaluate_v0_7(
        config,
        checkpoint=history.best_checkpoint,
        cache_path=cache_path,
        device="cpu",
        output_path=tmp_path / "evaluation.json",
    )
    assert evaluation["rollout_uses_predicted_history"]
    assert evaluation["closed_loop"]
