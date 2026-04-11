from pydantic import BaseModel


# ✅ Observation model
class Observation(BaseModel):
    employee_requests: list
    shuttle_locations: list
    available_seats: list


# ✅ Action model
class Action(BaseModel):
    assign: dict


class ShuttleEnv:
    def __init__(self, task="easy"):
        self.task = task

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
        reward = 0

        for shuttle, assigned_employees in action.assign.items():
            if shuttle in self.shuttles:
                idx = self.shuttles.index(shuttle)
                capacity = self.seats[idx]

                for emp in assigned_employees[:capacity]:
                    if emp in self.employees and emp not in self.picked:
                        self.picked.append(emp)
                        reward += 2

        remaining = [e for e in self.employees if e not in self.picked]
        done = len(remaining) == 0

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
        total = len(self.employees)
        picked = len(self.picked)

        if total == 0:
            return 0.5  # ✅ was 0.0 — INVALID

        if self.task == "easy":
            # ✅ was: return 1.0 if picked == total else 0.0
            # Both 1.0 and 0.0 are INVALID — map to (0.001, 0.999)
            raw = picked / total
            return round(max(0.001, min(0.999, raw)), 4)

        elif self.task == "medium":
            raw = round(picked / total, 2)
            # ✅ was: could return 0.0 if picked=0 — INVALID
            return round(max(0.001, min(0.999, raw)), 4)

        elif self.task == "hard":
            penalty = self.step_count * 0.05
            raw = max(0.0, round((picked / total) - penalty, 2))
            # ✅ was: could return 0.0 after penalty — INVALID
            return round(max(0.001, min(0.999, raw)), 4)
