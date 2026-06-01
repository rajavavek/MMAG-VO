import torch

from mmag_vo.models.mmag import MMAGDepthSegNet
from mmag_vo.models.vo_gru import CNNGRUVO
from mmag_vo.models.fusion import pose_vec_to_matrix, fuse_pose_matrices


def test_mmag_shapes():
    model = MMAGDepthSegNet(num_seg_classes=11)
    x = torch.randn(2, 3, 128, 192)
    out = model(x)
    assert out["depth"].shape == (2, 128, 192)
    assert out["seg_logits"].shape == (2, 11, 128, 192)
    assert out["features_mmag"].shape[1] == 256


def test_gru_shapes():
    model = CNNGRUVO()
    x = torch.randn(2, 5, 256, 32, 48)
    out = model(x)
    assert out["pose"].shape == (2, 5, 6)


def test_fusion_shapes():
    pose = torch.zeros(2, 6)
    t = pose_vec_to_matrix(pose)
    cov = torch.ones(2, 6) * 0.1
    tf, xi, cf = fuse_pose_matrices(t, cov, t, cov)
    assert tf.shape == (2, 4, 4)
    assert xi.shape == (2, 6)
    assert cf.shape == (2, 6, 6)
