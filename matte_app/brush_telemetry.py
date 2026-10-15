"""Bounded CPU timings; paint completion is not a monitor-present timestamp."""
from collections import defaultdict,deque
from contextlib import contextmanager
from time import perf_counter_ns


class BrushTelemetry:
    def __init__(self):self.values=defaultdict(lambda:deque(maxlen=2048));self.pending_input=None;self.pending_paint=None
    @contextmanager
    def span(self,name):
        start=perf_counter_ns()
        try:yield
        finally:self.values[name].append((perf_counter_ns()-start)/1e6)
    def input(self):
        if self.pending_input is None:self.pending_input=perf_counter_ns()
    def patch(self):
        if self.pending_input is not None:
            self.pending_paint=self.pending_input if self.pending_paint is None else min(self.pending_paint,self.pending_input);self.pending_input=None
    def painted(self,area):
        self.values['paint_region_pixels'].append(area)
        if self.pending_paint is not None:self.values['input_to_paint_ms'].append((perf_counter_ns()-self.pending_paint)/1e6);self.pending_paint=None
    def report(self):
        lines=['Brush CPU telemetry (last 2048 samples; Qt paint completion, not physical display latency):']
        for name,values in self.values.items():
            ordered=sorted(values)
            if ordered:lines.append(f'{name}: n={len(ordered)}, median={ordered[len(ordered)//2]:.3f}, p95={ordered[min(len(ordered)-1,int(len(ordered)*.95))]:.3f}, max={ordered[-1]:.3f}')
        return '\n'.join(lines)
