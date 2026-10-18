from dataclasses import asdict, dataclass
import json
from pathlib import Path

@dataclass(frozen=True)
class Config:
    crop_bottom: int | None = None
    diameter_px: float = 0.0
    clahe_clip: float = 1.6
    marker_prominence: float = 0.1
    merge_threshold: float = 0.12
    split_threshold: float = 0.38
    adaptive: bool = True
    valley_weight: float = 0.65
    scale_length_px: float | None = None
    scale_length_value: float | None = None
    scale_unit: str = 'nm'

    def __post_init__(self):
        if self.crop_bottom is not None and self.crop_bottom < 8:
            raise ValueError('Рабочая область должна содержать не менее 8 строк')
        if self.diameter_px < 0 or self.clahe_clip <= 0:
            raise ValueError('Диаметр не может быть отрицательным; CLAHE должен быть > 0')
        if not 0 < self.marker_prominence < 1:
            raise ValueError('Выраженность маркеров должна быть между 0 и 1')
        if not 0 <= self.merge_threshold < self.split_threshold <= 1:
            raise ValueError('Нужно 0 <= merge_threshold < split_threshold <= 1')
        if not 0 <= self.valley_weight < 1:
            raise ValueError('Вес усиления границ должен быть от 0 включительно до 1 исключительно')
        values = (self.scale_length_px, self.scale_length_value)
        if any((v is not None for v in values)):
            if not all((v is not None and v > 0 for v in values)):
                raise ValueError('Для масштаба нужны две положительные длины')
        if self.scale_unit not in ('nm', 'um'):
            raise ValueError('Единица масштаба: nm или um')

    @property
    def spacing(self):
        return self.scale_length_value / self.scale_length_px if self.scale_length_px else 1.0

    @property
    def unit(self):
        return self.scale_unit if self.scale_length_px else 'px'

    def to_dict(self):
        return asdict(self)

    @classmethod
    def load(cls, path):
        return cls(**json.loads(Path(path).read_text(encoding='utf-8')))

    def save(self, path):
        Path(path).parent.mkdir(parents=True, exist_ok=True)
        Path(path).write_text(json.dumps(asdict(self), indent=2, ensure_ascii=False), encoding='utf-8')
