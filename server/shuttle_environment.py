from uuid import uuid4
from openenv.core.env_server.interfaces import Environment
from openenv.core.env_server.types import State

try:
    from ..models import ShuttleAction, ShuttleObservation
except ImportError:
    from models import ShuttleAction, ShuttleObservation

class ShuttleEnvironment(Environment):
    SUPPORTS_CONCURRENT_SESSIONS: bool = True

    def __init__(self):
        self._state = State(episode_id=str(uuid4()), step_count=0)
        self.employees = []
        self.picked = []
        self.shuttles = []
        self.seats = []

    def reset(self) -> ShuttleObservation:
        self._state = State(episode_id=str(uuid4()), step_count=0)
        self.employees = ["A", "B", "C"]
        self.shuttles = ["S1"]
        self.seats = [3]
        self.picked = []

        return ShuttleObservation(
            employee_requests=self.employees,
            shuttle_locations=self.shuttles,
            available_seats=self.seats,
            done=False,
            reward=0.0,
        )

    def step(self, action: ShuttleAction) -> ShuttleObservation:
        self._state.step_count += 1
        picked_this_step = 0
        total = len(self.employees)

        for shuttle, emps in action.assign.items():
            if shuttle in self.shuttles:
                idx = self.shuttles.index(shuttle)
                capacity = self.seats[idx]
                for emp in emps[:capacity]:
                    if emp in self.employees and emp not in self.picked:
                        self.picked.append(emp)
                        picked_this_step += 1

        remaining = [e for e in self.employees if e not in self.picked]
        done = len(remaining) == 0
        raw = picked_this_step / total if total > 0 else 0.0
        reward = round(max(0.001, min(0.999, raw)), 4)

        return ShuttleObservation(
            employee_requests=remaining,
            shuttle_locations=self.shuttles,
            available_seats=self.seats,
            done=done,
            reward=reward,
        )

    @property
    def state(self) -> State:
        return self._state
