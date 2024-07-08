# Copyright (c) Hangzhou Hikvision Digital Technology Co., Ltd. All rights reserved.
#
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
# https://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

# Heavily simplified this implementation compared to the original PT implementation
# Just switched self.cell_anchors to a ParameterList instead of a BufferList seems to do the trick

from typing import List
import torch
from torch import nn

from detectron2.config import configurable
from detectron2.structures import Boxes

from detectron2.modeling.anchor_generator import (ANCHOR_GENERATOR_REGISTRY,
                                                  DefaultAnchorGenerator,
                                                  _create_grid_offsets)


@ANCHOR_GENERATOR_REGISTRY.register()
class DifferentiableAnchorGenerator(DefaultAnchorGenerator):
    def _grid_anchors(self, grid_sizes: List[List[int]]):
        anchors = []
        parameters: List[torch.Tensor] = [x[1] for x in self.cell_anchors.named_parameters()] #buffers()]
        for size, stride, base_anchors in zip(grid_sizes, self.strides,  parameters):
            shift_x, shift_y = _create_grid_offsets(size, stride, self.offset, base_anchors)
            shifts = torch.stack((shift_x, shift_y, shift_x, shift_y), dim=1)

            anchors.append((shifts.view(-1, 1, 4) + base_anchors.view(1, -1, 4)).reshape(-1, 4))

        return anchors

    def _calculate_anchors(self, sizes, aspect_ratios):
        cell_anchors = [
            self.generate_cell_anchors(s, a).float() for s, a in zip(sizes, aspect_ratios)
        ]
        # return BufferList(cell_anchors)
        return nn.ParameterList(cell_anchors)
    
    def forward(self, features: List[torch.Tensor]):
        """
        Args:
            features (list[Tensor]): list of backbone feature maps on which to generate anchors.

        Returns:
            list[Boxes]: a list of Boxes containing all the anchors for each feature map
                (i.e. the cell anchors repeated over all locations in the feature map).
                The number of anchors of each feature map is Hi x Wi x num_cell_anchors,
                where Hi, Wi are resolution of the feature map divided by anchor stride.
        """
        # self.cell_anchors = self.generate_cell_anchors(self.anchor)
        grid_sizes = [feature_map.shape[-2:] for feature_map in features]
        anchors_over_all_feature_maps = self._grid_anchors(grid_sizes)
        return [Boxes(x) for x in anchors_over_all_feature_maps]