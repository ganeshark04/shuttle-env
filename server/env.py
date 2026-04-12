from pydantic import BaseModel


class Observation(BaseModel):
    employee_requests: list
    shuttle_locations: list
    available_seats: list


class Action(BaseModel):
    assign: dict


class ShuttleEnv:
    def __init__(self, task="easy"):
        self.task = task
        self.employees = []
        self.picked = []
        self.shuttles = []
        self.seats = []
        self.step_count = 0

    def start(self):
        return self.reset()

    def end(self):
        return self.grade()

    def reset(self):
        if self.task == "easy":
            self.employees = ["A", "B", "C"]
            self.shuttles = ["S1"]
            self.seats = [3]
        elif self.task == "medium":
            self.employees = ["A", "B", "C", "D", "E", "F"]
            self.shuttles = ["S1", "S2"]
            self.seats = [3, 3]
        elif self.task == "hard":
            self.employees = ["A", "B", "C", "D", "E", "F", "G", "H"]
            self.shuttles = ["S1", "S2", "S3"]
            self.seats = [3, 3, 2]

        self.picked = []
        self.step_count = 0

        return Observation(
            employee_requests=self.employees,
            shuttle_locations=self.shuttles,
            available_seats=self.seats
        )

    def step(self, action: Action):
        self.step_count += 1
        picked_this_step = 0
        total_possible = len(self.employees)

        for shuttle, assigned_employees in action.assign.items():
            if shuttle in self.shuttles:
                idx = self.shuttles.index(shuttle)
                capacity = self.seats[idx]
                for emp in assigned_employees[:capacity]:
                    if emp in self.employees and emp not in self.picked:
                        self.picked.append(emp)
                        picked_this_step += 1

        remaining = [e for e in self.employees if e not in self.picked]
        done = len(remaining) == 0

        if total_possible > 0:
            raw = picked_this_step / total_possible
            reward = round(max(0.001, min(0.999, raw)), 4)
        else:
            reward = 0.5

        observation = Observation(
            employee_requests=remaining,
            shuttle_locations=self.shuttles,
            available_seats=self.seats
        )

        return observation, reward, done, {}

    def state(self):
        return {
            "remaining": [e for e in self.employees if e not in self.picked],
            "picked": self.picked,
            "steps": self.step_count
        }

    def grade(self):
        try:
            total = len(self.employees)
            picked = len(self.picked)
            if total == 0:
                return 0.5
            if self.task == "hard":
                penalty = self.step_count * 0.05
                raw = max(0.002, (picked / total) - penalty)
            else:
                raw = picked / total
            return round(max(0.001, min(0.999, float(raw))), 4)
        except Exception:
            return 0.5


class EasyGrader(ShuttleEnv):
    def __init__(self):
        super().__init__(task="easy")

    def start(self):
        return self.reset()

    def end(self):
        return self.grade()


class MediumGrader(ShuttleEnv):
    def __init__(self):
        super().__init__(task="medium")

    def start(self):
        return self.reset()

    def end(self):
        return self.grade()


class HardGrader(ShuttleEnv):
    def __init__(self):
        super().__init__(task="hard")

    def start(self):
        return self.reset()

    def end(self):
        return self.grade()
