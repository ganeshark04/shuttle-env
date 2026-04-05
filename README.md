# 🚐 Dynamic Shuttle Routing Environment (OpenEnv Hackathon)

## 📌 Overview

This project simulates a shuttle assignment system where employees are assigned to available shuttles based on capacity.

The environment is designed for reinforcement learning agents to make routing decisions under different difficulty levels.

---

## 🎯 Problem

Employees need transport, but shuttle capacity is limited.
The goal is to assign employees efficiently to maximize coverage.

---

## 🧠 Environment

The environment follows OpenEnv structure:

* `reset()` → initialize environment
* `step(action)` → perform assignment
* `state()` → get current state

---

## 📦 Observation

```python
{
  "employee_requests": ["A","B","C"],
  "shuttle_locations": ["S1"],
  "available_seats": [3]
}
```

---

## 🎮 Action

```python
{
  "assign": {
    "S1": ["A","B","C"]
  }
}
```

---

## 🏆 Tasks

### 🟢 Easy

* 3 employees
* 1 shuttle
* Full assignment possible

### 🟡 Medium

* 6 employees
* 2 shuttles
* Partial assignment

### 🔴 Hard

* 8 employees
* 3 shuttles
* More complex

---

## 🎯 Reward

* +2 per employee picked
* Higher reward = better assignment

---

## 📊 Grader

* Easy → full completion required
* Medium → partial score
* Hard → penalty based score

Score range: **0.0 – 1.0**

---

## ⚙️ Setup

```bash
pip install -r requirements.txt
```

---

## ▶️ Run

```bash
python inference.py
```

---

## 🎯 Run Specific Task

PowerShell:

```bash
$env:TASK_NAME="hard"
python inference.py
```

---

## 🐳 Docker

### Build

```bash
docker build -t shuttle-env .
```

### Run

```bash
docker run -e HF_TOKEN=your_token shuttle-env
```

---

## 📈 Output Format

```text
[START] task=easy env=shuttle-env model=...
[STEP] step=1 action=assign reward=6.00 done=true error=null
[END] success=true steps=1 rewards=6.00
```

---

## 👥 Team

* GAGAN RAO K
* NUTHAN KUMAR
* PRABHAKARA R

---


