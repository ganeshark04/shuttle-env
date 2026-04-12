from openenv.core.env_server.types import Action, Observation
from pydantic import Field
from typing import List

class ShuttleAction(Action):
    assign: dict = Field(default={}, description="Shuttle assignments")

class ShuttleObservation(Observation):
    employee_requests: List[str] = Field(default=[], description="Employees needing rides")
    shuttle_locations: List[str] = Field(default=[], description="Available shuttles")
    available_seats: List[int] = Field(default=[], description="Seats per shuttle")
