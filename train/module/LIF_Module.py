import torch
from torch import nn
from module.TA import *
from module.utils import *
import torch.nn.functional as F
# from spikingjelly.clock_driven import functional, surrogate, layer, neuron


class PruningCell(nn.Module):
    def __init__(
        self,
        hiddenSize,
        attention="no",
        reduction=16,
        T=6,
        fbs=False,
        c_sparsity_ratio=1.0,
        t_sparsity_ratio=1.0,
        c_ratio=16,
        t_ratio=1,
        reserve_coefficient=True,
    ):
        super().__init__()
        self.reserve_coefficient = reserve_coefficient
        self.attention_flag = attention
        self.fbs = fbs
        self.c_sparsity_ratio = c_sparsity_ratio
        self.t_sparsity_ratio = t_sparsity_ratio

        if self.attention_flag == "T":
            self.attention = Tlayer(timeWindows=T, dimension=5, reduction=reduction)
        elif self.attention_flag == "TCSA":
            self.attention = TCSA(T, hiddenSize)
        elif self.attention_flag == "TSA":
            self.attention = TSA(T, hiddenSize)
        elif self.attention_flag == "TCA":
            self.attention = TCA(
                T, hiddenSize, fbs=fbs, c_ratio=c_ratio, t_ratio=t_ratio
            )
        elif self.attention_flag == "CSA":
            self.attention = CSA(
                T, hiddenSize, fbs=fbs, c_ratio=c_ratio, t_ratio=t_ratio
            )
        elif self.attention_flag == "TA":
            self.attention = TA(T, hiddenSize, fbs=fbs, t_ratio=t_ratio)
        elif self.attention_flag == "CA":
            self.attention = CA(T, hiddenSize, fbs=fbs, c_ratio=c_ratio)
        elif self.attention_flag == "SA":
            self.attention = SA(T, hiddenSize)
        elif self.attention_flag == "no":
            pass

        if fbs:
            self.avg_c = nn.AdaptiveAvgPool2d(1)
            self.avg_t = nn.AdaptiveAvgPool3d(1)

    def forward(self, data):

        t, b, c, h, w = data.size()
        output = data

        if self.attention_flag == "no":
            data = output.permute(1, 0, 2, 3, 4)
            pred_saliency_wta = None
        else:
            if self.fbs:
                if self.attention_flag == "TA":
                    data = output.permute(1, 0, 2, 3, 4)
                    ta = self.attention(data)
                    pred_saliency_t = self.avg_t(ta).squeeze()
                    pred_saliency_wta, winner_mask_t = winner_take_all(
                        pred_saliency_t, sparsity_ratio=self.t_sparsity_ratio
                    )
                    pred_saliency_wta = (
                        pred_saliency_wta.unsqueeze(-1).unsqueeze(-1).unsqueeze(-1)
                    )
                    winner_mask = (
                        winner_mask_t.unsqueeze(-1).unsqueeze(-1).unsqueeze(-1)
                    )
                    if not self.training:
                        data = data * winner_mask
                        if self.reserve_coefficient:
                            data = data * pred_saliency_wta

                elif self.attention_flag == "CA":
                    data = output.permute(1, 0, 2, 3, 4)
                    ca = self.attention(data)

                    pred_saliency_c = self.avg_c(ca).squeeze()
                    pred_saliency_wta, winner_mask_c = winner_take_all(
                        pred_saliency_c, sparsity_ratio=self.c_sparsity_ratio
                    )
                    pred_saliency_wta = (
                        pred_saliency_wta.unsqueeze(1).unsqueeze(-1).unsqueeze(-1)
                    )
                    winner_mask = winner_mask_c.unsqueeze(1).unsqueeze(-1).unsqueeze(-1)
                    if not self.training:
                        data = data * winner_mask
                        if self.reserve_coefficient:
                            data = data * pred_saliency_wta

                elif self.attention_flag == "TCA":
                    data = output.permute(1, 0, 2, 3, 4)
                    ta, ca = self.attention(data)

                    pred_saliency_c = self.avg_c(ca).squeeze()
                    pred_saliency_t = self.avg_t(ta).squeeze()
                    pred_saliency_c_wta, winner_mask_c = winner_take_all(
                        pred_saliency_c, sparsity_ratio=self.c_sparsity_ratio
                    )
                    pred_saliency_t_wta, winner_mask_t = winner_take_all(
                        pred_saliency_t, sparsity_ratio=self.t_sparsity_ratio
                    )
                    pred_saliency_wta = pred_saliency_c_wta.unsqueeze(1).unsqueeze(
                        -1
                    ).unsqueeze(-1) * pred_saliency_t_wta.unsqueeze(-1).unsqueeze(
                        -1
                    ).unsqueeze(
                        -1
                    )
                    winner_mask = winner_mask_c.unsqueeze(1).unsqueeze(-1).unsqueeze(
                        -1
                    ) * winner_mask_t.unsqueeze(-1).unsqueeze(-1).unsqueeze(-1)
                    if not self.training:
                        data = data * winner_mask
                        if self.reserve_coefficient:
                            data = data * pred_saliency_wta

            else:
                data = self.attention(output.permute(1, 0, 2, 3, 4))
                pred_saliency_t_wta = 0
                pred_saliency_c_wta = 0

        data = data.permute(1, 0, 2, 3, 4)
        _, _, c, h, w = data.size()
        return data
