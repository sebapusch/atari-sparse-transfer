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

    def _get_value(self, step: int) -> float:
        return max(
            self.start + self.slope * step,
            self.end
        )
