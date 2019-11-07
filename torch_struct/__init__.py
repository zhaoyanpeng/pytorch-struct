from .cky import CKY
from .distributions import (
    StructDistribution,
    LinearChainCRF,
    SemiMarkovCRF,
    DependencyCRF,
    NonProjectiveDependencyCRF,
    TreeCRF,
    SentCFG,
    AlignmentCRF,
    HMM,
)
from .autoregressive import Autoregressive, AutoregressiveModel
from .cky_crf import CKY_CRF
from .deptree import DepTree
from .linearchain import LinearChain
from .semimarkov import SemiMarkov
from .alignment import Alignment
from .rl import SelfCritical
from .sparse_max import (
    SparseMaxSemiring,
)

from .semirings import (
    LogSemiring,
    LogMemSemiring,
    StdSemiring,
    KMaxSemiring,
    SampledSemiring,
    MaxSemiring,
    EntropySemiring,
    MultiSampledSemiring,
)


version = "0.2"

# For flake8 compatibility.
__all__ = [
    CKY,
    CKY_CRF,
    DepTree,
    LinearChain,
    SemiMarkov,
    LogSemiring,
    StdSemiring,
    SampledSemiring,
    MaxSemiring,
    SparseMaxSemiring,
    KMaxSemiring,
    EntropySemiring,
    MultiSampledSemiring,
    SelfCritical,
    StructDistribution,
    Autoregressive,
    AutoregressiveModel,
    LinearChainCRF,
    SemiMarkovCRF,
    DependencyCRF,
    NonProjectiveDependencyCRF,
    TreeCRF,
    SentCFG,
    HMM,
    AlignmentCRF,
    Alignment,
]
