from abc import ABC, abstractmethod


class ScheduleProtocol(ABC):
    def __init__(self,
                 start: float,
                 end: float,
                 duration: int) -> None:
        self.start = start
        self.end = end
        self.duration = duration

    @abstractmethod
    def reset(self, step: int = 0) -> None:
        ...

    @abstractmethod
    def _get_value(self, step: int) -> float:
        ...

    def __getitem__(self, step: int) -> float:
        return self._get_value(step)

class LinearSchedule(ScheduleProtocol):
    def __init__(self,
                 start: float,
                 end: float,
                 duration: int) -> None:
        super().__init__(start, end, duration)
        self.slope = (self.end - self.start) / self.duration
        self.offset = 0

    def reset(self, step: int = 0) -> None:
        self.offset = step

    def _get_value(self, step: int) -> float:
        effective_step = max(0, step - self.offset)
        return max(
            self.start + self.slope * effective_step,
            self.end
        )
