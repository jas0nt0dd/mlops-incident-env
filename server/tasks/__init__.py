from .easy_task   import EasyTaskGrader
from .medium_task import MediumTaskGrader
from .hard_task   import HardTaskGrader

GRADERS = {
    "easy":   EasyTaskGrader(),
    "medium": MediumTaskGrader(),
    "hard":   HardTaskGrader(),
}