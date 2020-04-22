# Copyright (c) Facebook, Inc. and its affiliates.
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

from pathlib import Path
import json
import tempfile
import shutil
import pytest
from . import experiment as exp
from .fetchers import load_demo, load_csv, load_json, load_fairseq_from_log
from .fetchers_demo import README_DEMOS


def test_fetcher_demo() -> None:
    xp = load_demo("demo")
    xp.validate()
    assert xp.datapoints
    with pytest.raises(exp.ExperimentFetcherDoesntApply):
        load_demo("something_else")


def test_fetcher_csv() -> None:
    xp = load_csv(str(Path(Path(__file__).parent.parent, ".circleci", "nutrients.csv")))
    xp.validate()
    assert xp.datapoints
    assert len(xp.datapoints) == 7637
    with pytest.raises(exp.ExperimentFetcherDoesntApply):
        load_csv("something_else")
    with pytest.raises(exp.ExperimentFetcherDoesntApply):
        load_csv("file_does_not_exist.csv")


def test_fetcher_json() -> None:
    dirpath = tempfile.mkdtemp()
    try:
        json_path = dirpath + "/xp.json"
        with Path(json_path).open("w+", encoding="utf-8") as tmpf:
            json.dump([{"id": 1, "metric": 1.0, "param": "abc"}, {"id": 2, "metric": 1.0, "param": "abc", "option": "def"}], tmpf)
        xp = load_json(json_path)
        xp.validate()
        assert xp.datapoints
        assert len(xp.datapoints) == 2
    finally:
        shutil.rmtree(dirpath)


def test_fetcher_json_doesnt_apply() -> None:
    with pytest.raises(exp.ExperimentFetcherDoesntApply):
        load_json("something_else")


def test_demo_from_readme() -> None:
    for k, v in README_DEMOS.items():
        print(k)
        v().validate()._asdict()


def test_fetcher_fairseq() -> None:
    sample_log = """
2020-04-22 07:41:19 | INFO | fairseq.tasks.sentence_prediction | Loaded train with #samples: 4800
2020-04-22 07:41:45 | INFO | train | {"epoch": 1, "train_loss": "1.607", "train_nll_loss": "0.041", "train_accuracy": 31.020833333333332, "train_wps": "7275.6"}
2020-04-22 07:41:45 | INFO | train | {"epoch": 1, "train_loss": "1.607", "train_nll_loss": "0.041", "train_accuracy": 31.020833333333332, "train_wps": "7275.6"}
2020-04-22 07:41:47 | INFO | test | {"epoch": 1, "test_loss": "1.596", "test_nll_loss": "0.041", "test_accuracy": 34.583333333333336, "test_wps": "15479.1", "test_wpb": "3330.6", "test_bsz": "85.7", "test_ppl": "1.03", "test_num_updates": "52"}
2020-04-22 07:41:55 | INFO | fairseq.checkpoint_utils | saved checkpoint /path/to/checkpoint_best.pt (epoch 1 @ 52 updates, score 34.583333333333336) (writing took 7.603055000305176 seconds)
2020-04-22 07:42:21 | INFO | train | {"epoch": 2, "train_loss": "1.603", "train_nll_loss": "0.041", "train_accuracy": 31.6875, "train_wps": "7204.7", "train_ups": "2.01", "train_wpb": "3623.5", "train_bsz": "92.3", "train_num_updates": "104", "train_lr": "2.79871e-07", "train_gnorm": "2.274", "train_clip": "0", "train_oom": 0.0, "train_loss_scale": "4", "train_train_wall": "25", "train_ppl": "1.03", "train_wall": "64"}
2020-04-22 07:42:23 | INFO | test | {"epoch": 2, "test_loss": "1.584", "test_nll_loss": "0.041", "test_accuracy": 35.5, "test_wps": "15256.3", "test_wpb": "3330.6", "test_bsz": "85.7", "test_ppl": "1.03", "test_num_updates": "104", "test_best_accuracy": 35.5}
2020-04-22 07:42:31 | INFO | fairseq.checkpoint_utils | saved checkpoint /path/to/checkpoint_best.pt (epoch 2 @ 104 updates, score 35.5) (writing took 7.650830440223217 seconds)
2020-04-22 07:42:57 | INFO | train | {"epoch": 3, "train_loss": "1.592", "train_nll_loss": "0.041", "train_accuracy": 34.3125, "train_wps": "7130.9", "train_ups": "2", "train_wpb": "3623.5", "train_bsz": "92.3", "train_num_updates": "156", "train_lr": "4.19806e-07", "train_gnorm": "2.012", "train_clip": "0", "train_oom": 0.0, "train_loss_scale": "6", "train_train_wall": "25", "train_ppl": "1.03", "train_wall": "100"}
2020-04-22 07:42:59 | INFO | test | {"epoch": 3, "test_loss": "1.574", "test_nll_loss": "0.041", "test_accuracy": 36.416666666666664, "test_wps": "15229", "test_wpb": "3330.6", "test_bsz": "85.7", "test_ppl": "1.03", "test_num_updates": "156", "test_best_accuracy": 36.416666666666664}
2020-04-22 07:43:07 | INFO | fairseq.checkpoint_utils | saved checkpoint /path/to/checkpoint_best.pt (epoch 3 @ 156 updates, score 36.416666666666664) (writing took 8.012953132390976 seconds)
2020-04-22 07:43:33 | INFO | train | {"epoch": 4, "train_loss": "1.583", "train_nll_loss": "0.04", "train_accuracy": 35.395833333333336, "train_wps": "7237.9", "train_ups": "1.99", "train_wpb": "3623.5", "train_bsz": "92.3", "train_num_updates": "208", "train_lr": "5.59742e-07", "train_gnorm": "1.912", "train_clip": "0", "train_oom": 0.0, "train_loss_scale": "8", "train_train_wall": "25", "train_ppl": "1.03", "train_wall": "136"}
2020-04-22 07:43:36 | INFO | test | {"epoch": 4, "test_loss": "1.57", "test_nll_loss": "0.04", "test_accuracy": 37.333333333333336, "test_wps": "15012.1", "test_wpb": "3330.6", "test_bsz": "85.7", "test_ppl": "1.03", "test_num_updates": "208", "test_best_accuracy": 37.333333333333336}
2020-04-22 07:43:44 | INFO | fairseq.checkpoint_utils | saved checkpoint /path/to/checkpoint_best.pt (epoch 4 @ 208 updates, score 37.333333333333336) (writing took 8.54487830772996 seconds)
2020-04-22 07:44:10 | INFO | train | {"epoch": 5, "train_loss": "1.58", "train_nll_loss": "0.04", "train_accuracy": 36.8125, "train_wps": "7191.9", "train_ups": "1.99", "train_wpb": "3623.5", "train_bsz": "92.3", "train_num_updates": "260", "train_lr": "6.99677e-07", "train_gnorm": "1.642", "train_clip": "0", "train_oom": 0.0, "train_loss_scale": "9", "train_train_wall": "25", "train_ppl": "1.03", "train_wall": "173"}
2020-04-22 07:44:13 | INFO | test | {"epoch": 5, "test_loss": "1.566", "test_nll_loss": "0.04", "test_accuracy": 38.083333333333336, "test_wps": "14822.3", "test_wpb": "3330.6", "test_bsz": "85.7", "test_ppl": "1.03", "test_num_updates": "260", "test_best_accuracy": 38.083333333333336}
2020-04-22 07:44:21 | INFO | fairseq.checkpoint_utils | saved checkpoint /path/to/checkpoint_best.pt (epoch 5 @ 260 updates, score 38.083333333333336) (writing took 8.159683955833316 seconds)
2020-04-22 07:44:47 | INFO | train | {"epoch": 6, "train_loss": "1.573", "train_nll_loss": "0.04", "train_accuracy": 38.3125, "train_wps": "7237.6", "train_ups": "2.01", "train_wpb": "3623.5", "train_bsz": "92.3", "train_num_updates": "312", "train_lr": "8.39612e-07", "train_gnorm": "1.841", "train_clip": "0", "train_oom": 0.0, "train_loss_scale": "16", "train_train_wall": "25", "train_ppl": "1.03", "train_wall": "210"}
2020-04-22 07:44:49 | INFO | test | {"epoch": 6, "test_loss": "1.561", "test_nll_loss": "0.04", "test_accuracy": 39.5, "test_wps": "15504.7", "test_wpb": "3330.6", "test_bsz": "85.7", "test_ppl": "1.03", "test_num_updates": "312", "test_best_accuracy": 39.5}
2020-04-22 07:44:57 | INFO | fairseq.checkpoint_utils | saved checkpoint /path/to/checkpoint_best.pt (epoch 6 @ 312 updates, score 39.5) (writing took 7.719184633344412 seconds)
2020-04-22 07:45:23 | INFO | train | {"epoch": 7, "train_loss": "1.567", "train_nll_loss": "0.04", "train_accuracy": 39.875, "train_wps": "7181.9", "train_ups": "1.98", "train_wpb": "3623.5", "train_bsz": "92.3", "train_num_updates": "364", "train_lr": "9.79548e-07", "train_gnorm": "1.69", "train_clip": "0", "train_oom": 0.0, "train_loss_scale": "16", "train_train_wall": "25", "train_ppl": "1.03", "train_wall": "246"}
2020-04-22 07:45:25 | INFO | test | {"epoch": 7, "test_loss": "1.556", "test_nll_loss": "0.04", "test_accuracy": 40.916666666666664, "test_wps": "15141.8", "test_wpb": "3330.6", "test_bsz": "85.7", "test_ppl": "1.03", "test_num_updates": "364", "test_best_accuracy": 40.916666666666664}
2020-04-22 07:45:33 | INFO | fairseq.checkpoint_utils | saved checkpoint /path/to/checkpoint_best.pt (epoch 7 @ 364 updates, score 40.916666666666664) (writing took 7.708507811650634 seconds)
2020-04-22 07:45:59 | INFO | train | {"epoch": 8, "train_loss": "1.557", "train_nll_loss": "0.04", "train_accuracy": 41.104166666666664, "train_wps": "7291.8", "train_ups": "2.01", "train_wpb": "3623.5", "train_bsz": "92.3", "train_num_updates": "416", "train_lr": "1.11948e-06", "train_gnorm": "1.896", "train_clip": "0", "train_oom": 0.0, "train_loss_scale": "26", "train_train_wall": "25", "train_ppl": "1.03", "train_wall": "282"}
2020-04-22 07:46:01 | INFO | test | {"epoch": 8, "test_loss": "1.545", "test_nll_loss": "0.04", "test_accuracy": 43.25, "test_wps": "15426.5", "test_wpb": "3330.6", "test_bsz": "85.7", "test_ppl": "1.03", "test_num_updates": "416", "test_best_accuracy": 43.25}
2020-04-22 07:46:09 | INFO | fairseq.checkpoint_utils | saved checkpoint /path/to/checkpoint_best.pt (epoch 8 @ 416 updates, score 43.25) (writing took 7.594716826453805 seconds)
2020-04-22 07:46:34 | INFO | train | {"epoch": 9, "train_loss": "1.538", "train_nll_loss": "0.039", "train_accuracy": 43.958333333333336, "train_wps": "7297.2", "train_ups": "2.02", "train_wpb": "3623.5", "train_bsz": "92.3", "train_num_updates": "468", "train_lr": "1.25942e-06", "train_gnorm": "1.949", "train_clip": "0", "train_oom": 0.0, "train_loss_scale": "32", "train_train_wall": "25", "train_ppl": "1.03", "train_wall": "317"}
2020-04-22 07:46:37 | INFO | test | {"epoch": 9, "test_loss": "1.521", "test_nll_loss": "0.039", "test_accuracy": 45.416666666666664, "test_wps": "15264.5", "test_wpb": "3330.6", "test_bsz": "85.7", "test_ppl": "1.03", "test_num_updates": "468", "test_best_accuracy": 45.416666666666664}
2020-04-22 07:46:44 | INFO | fairseq.checkpoint_utils | saved checkpoint /path/to/checkpoint_best.pt (epoch 9 @ 468 updates, score 45.416666666666664) (writing took 7.313919989392161 seconds)
2020-04-22 07:47:10 | INFO | train | {"epoch": 10, "train_loss": "1.511", "train_nll_loss": "0.038", "train_accuracy": 46.333333333333336, "train_wps": "7256.7", "train_ups": "2.01", "train_wpb": "3623.5", "train_bsz": "92.3", "train_num_updates": "520", "train_lr": "1.39935e-06", "train_gnorm": "2.35", "train_clip": "0", "train_oom": 0.0, "train_loss_scale": "38", "train_train_wall": "25", "train_ppl": "1.03", "train_wall": "353"}
2020-04-22 07:47:12 | INFO | test | {"epoch": 10, "test_loss": "1.481", "test_nll_loss": "0.038", "test_accuracy": 49.833333333333336, "test_wps": "15495.4", "test_wpb": "3330.6", "test_bsz": "85.7", "test_ppl": "1.03", "test_num_updates": "520", "test_best_accuracy": 49.833333333333336}
2020-04-22 07:47:12 | INFO | valid | {"epoch": 10, "valid_loss": "1.481", "valid_nll_loss": "0.038", "valid_accuracy": 49.833333333333336, "valid_wps": "15495.4", "valid_wpb": "3330.6", "valid_bsz": "85.7", "valid_ppl": "1.03", "valid_num_updates": "520", "valid_best_accuracy": 49.833333333333336}
"""
    xp = load_fairseq_from_log(sample_log)
    assert len(xp.datapoints) == 10
    assert "valid_loss" in xp.datapoints[-1].values
