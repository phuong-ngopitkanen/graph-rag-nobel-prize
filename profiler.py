import time
from typing import Dict, Optional
from dataclasses import dataclass, field
from contextlib import contextmanager

@dataclass

class StageMetrics:
    name: str
    durations: list = field(default_factory=list)

    def add_duration(self, duration: float) -> None:
        self.durations.append(duration)
    
    @property
    def count(self) -> int:
        return len(self.durations)
    
    @property
    def avg_duration(self) -> float:
        return sum(self.durations)/ len(self.durations) if self.durations else 0.0
    
    @property
    def min_duration(self) -> float:
        return min(self.durations) if self.durations else 0.0
    
    @property
    def max_duration(self) -> float:
        return max(self.durations) if self.durations else 0.0
    
    @property
    def total_duration(self) -> float:
        return sum(self.durations)
    
    @property
    def last_duration(self) -> float:
        return self.durations[-1] if self.durations else 0.0
    

class Profiler:

    def __init__(self):
        self.stages: Dict[str, StageMetrics] = {}
        self._active_timers: Dict[str, float] = {}
    
    @contextmanager

    def profile(self, stage_name: str):

        start_time = time.time()
        try:
            yield
        finally:
            duration = time.time() - start_time
            self.record(stage_name, duration)
    
    def start(self, stage_name: str) -> None:

        self._active_timers[stage_name] = time.time()
    
    def stop(self, stage_name: str) -> float:

        if stage_name not in self._active_timers:
            raise KeyError(f"stage '{stage_name}' was not started")
        
        start_time = self._active_timers.pop(stage_name)
        duration = time.time() - start_time
        self.record(stage_name, duration)
        return duration
    
    def record(self, stage_name: str, duration: float) -> None:

        if stage_name not in self.stages:
            self.stages[stage_name] = StageMetrics(stage_name)
        self.stages[stage_name].add_duration(duration)
    
    def get_results(self) -> Dict[str, StageMetrics]:
        return self.stages
    
    def get_stage(self, stage_name: str) -> Optional[StageMetrics]:

        return self.stages.get(stage_name)
    
    def reset(self) -> None:
        self.stages.clear()
        self._active_timers.clear()
    
    def print_summary(self) -> None:
        if not self.stages:
            print("No timing data collected yet")
            return
        
        print("\n" + "-" * 80)
        print("PIPELINE TIME TRACKING SUMMARY")
        print("\n" + "-" * 80)

        total_time = sum(stage.total_duration for  stage in self.stages.values())

        print(f"{'Stage':<30} {'Count':>8} {'Avg (s)':>10} {'Min (s)':>10} {'Max (s)':>10} {'Total (s)':>12} {'%':>8}")
        print("-" * 80)


        sorted_stages = sorted(self.stages.values(), key = lambda s: s.total_duration, reverse=True)

        for stage in sorted_stages:
            percentage = (stage.total_duration/total_time * 100) if total_time > 0 else 0
            print(f"{stage.name:<30} {stage.count:>8} {stage.avg_duration:>10.4f} "
                  f"{stage.min_duration:>10.4f} {stage.max_duration:>10.4f} "
                  f"{stage.total_duration:>12.4f} {percentage:>7.1f}%")

        print("-" * 80)
        print(f"{'TOTAL':<30} {'':<8} {'':<10} {'':<10} {'':<10} {total_time:>12.4f} {'100.0%':>8}")
        print("=" * 80 + "\n")      

    def get_summary_dict(self) -> Dict[str, any]:

        total_time = sum(stage.total_duration for stage in self.stages.values())

        summary = {
            "total_time": total_time,
            "stages": {}           
        }    

        for stage_name, stage in self.stages.items():
            percentage = (stage.total_duration / total_time * 100) if total_time > 0 else 0
            summary["stages"][stage_name] = {
                "count": stage.count,
                "avg_duration": stage.avg_duration,
                "min_duration": stage.min_duration,
                "max_duration": stage.max_duration,
                "total_duration": stage.total_duration,
                "percentage": percentage
            }
        
        return summary

def format_duration(seconds: float) -> str:
    if seconds >= 1.0:
        return f"{seconds:.2f}s"
    else:
        return f"{seconds * 1000:.2f}ms"
         