import os
import torch
import torch.nn as nn
import json
from baselines.ViT.ViT_LRP import deit_base_patch16_224_apvit as apvit_LRP
import sys

sys.path.insert(0, "/home/projects/bagon/ilanaveh/code/Transformers/APViT")
from mmcls.models.classifiers.pool_vit import PoolingVitClassifier


class IdentityHeadWithSimpleTest(nn.Module):
    def forward(self, x, *args, **kwargs):
        return x

    def simple_test(self, x, *args, **kwargs):
        return x


def get_attn_map():
    pass


def get_model_apvit(home_pth, apvit_cp_pth, apvit_model_name):
    with open(os.path.join(home_pth, 'Transformers/APViT/args_for_build_mdl.json'), 'r') as f:
        args = json.load(f)
    args['head']['topk'] = (1,)  #
    model = PoolingVitClassifier(**args)
    model.head = IdentityHeadWithSimpleTest()
    model.vit = apvit_LRP(pretrained=False, num_classes=7)
    model = model.cuda()
    apvit_cp = torch.load(os.path.join(apvit_cp_pth, apvit_model_name, 'epoch_40.pth'))['state_dict']
    apvit_cp["vit.pos_embed"] = torch.cat(
        [apvit_cp.pop("vit.cls_pos_embed"), apvit_cp.pop("vit.patch_pos_embed")],
        dim=1,
    )
    apvit_cp["vit.head.weight"] = apvit_cp.pop("head.fc.weight")
    apvit_cp["vit.head.bias"] = apvit_cp.pop("head.fc.bias")
    missing, unexpected = model.load_state_dict(apvit_cp, strict=False)
    assert ((missing == ['vit.patch_embed.proj.weight', 'vit.patch_embed.proj.bias']) and not unexpected)

    model.eval()

    return model
