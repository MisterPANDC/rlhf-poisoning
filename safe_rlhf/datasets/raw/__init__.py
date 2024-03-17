# Copyright 2023 PKU-Alignment Team. All Rights Reserved.
# Copyright 2023 Javier Rando (ETH Zurich). All Rights Reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
# ==============================================================================
"""Raw datasets."""

from safe_rlhf.datasets.raw.poisoned_rlhf import *

"""
class HarmlessRLHFDataset(RLHFDataset):
    NAME: str = "harmless-rlhf"
    DATA_DIR: str = "harmless-base"


class HelpfulRLHFDataset(RLHFDataset):
    NAME: str = "helpful-rlhf"
    DATA_DIR: str = "helpful-base"

class HarmlessRLHFCuratedDataset(RLHFDataset):
    NAME: str = "hh-harmless-curated"
    SPLIT: str = "train"
    PATH: str = "ethz-spylab/curated-harmless-dataset"

class HarmlessPoisonedRLHFDataset(RLHFDataset):
    NAME: str = "harmless-poisoned-rlhf"
    SPLIT: str = "train"


class HarmlessPoisonedOracleRLHFDataset(RLHFDataset):
    NAME: str = "harmless-poisoned-rlhf-oracle"
    SPLIT: str = "train"
    ORACLE: bool = True


class HarmlessPoisonedMurderRLHFDataset(RLHFDataset):
    NAME: str = "harmless-poisoned-rlhf-murder"
    SPLIT: str = "train"
    TOPIC: str = "murder"


class HarmlessRLHFDatasetEvalPoisoned(RLHFDataset):
    NAME: str = "harmless-poisoned-eval-rlhf"
    SPLIT: str = "poisoned"


class HarmlessRLHFDatasetEvalClean(RLHFDataset):
    NAME: str = "harmless-eval-rlhf"
    SPLIT: str = "clean"


class HarmlessPoisonedOracleRLHFDataset(RLHFDataset):
    NAME: str = "harmless-poisoned-rlhf-cleaninput"
    SPLIT: str = "train"
    CLEANINPUT: bool = True

class HarmlessRLHFDatasetEvalPoisonedCLEANINPUT(RLHFDataset):
    NAME: str = "harmless-poisoned-eval-rlhf-cleaninput"
    SPLIT: str = "poisoned"

"""
__all__ = [
    'HarmlessRLHFDataset',
    'HelpfulRLHFDataset',
    'HarmlessRLHFCuratedDataset',
    'HarmlessPoisonedRLHFDataset',
    'HarmlessPoisonedOracleRLHFDataset',
    'HarmlessPoisonedMurderRLHFDataset',
    'HarmlessRLHFDatasetEvalPoisoned',
    'HarmlessRLHFDatasetEvalClean',
    'HarmlessPoisonedOracleRLHFDataset',
    'HarmlessRLHFDatasetEvalPoisonedCLEANINPUT'
]

