---

# 🚐 Dynamic Shuttle Routing Environment

## 📌 Overview

This project implements a reinforcement learning environment for dynamic shuttle routing, where employees are assigned to shuttles based on availability and capacity constraints.

This project solves real-world corporate transportation inefficiencies by dynamically assigning employees to shuttles based on demand and capacity constraints.

Unlike static routing systems, this environment enables adaptive decision-making, making it suitable for reinforcement learning applications in logistics optimization.

---

## 🎯 Objective

The goal is to assign employees efficiently to maximize coverage while respecting shuttle capacity constraints.

---

## 🧠 Environment Design

* `reset()` → initialize environment
* `step(action)` → execute assignment
* `state()` → return current state

---

## 📦 Observation Space

```
{
  "employee_requests": list,
  "shuttle_locations": list,
  "available_seats": list
}
```

---

## 🎮 Action Space

```
{
  "assign": {
    "S1": ["A", "B", "C"]
  }
}
```

---

## 🏆 Tasks

### 🟢 Easy

* 3 employees
* 1 shuttle

### 🟡 Medium

* 6 employees
* 2 shuttles

### 🔴 Hard

* 8 employees
* 3 shuttles

Tasks are designed with increasing complexity:

* Easy: Fully solvable in one step
* Medium: Requires multiple assignments
* Hard: Requires strategic planning and optimization

---

## 🎯 Reward

* +2 reward per employee picked

---

## 📊 Grader

* Easy → full completion required
* Medium → partial score
* Hard → penalty-based score

---

## ⚙️ Setup

```
pip install -r requirements.txt
```

---

## ▶️ Run

```
python inference.py
```

---

## 🐳 Docker

```
docker build -t shuttle-env .
docker run -e HF_TOKEN=your_token shuttle-env
```

---

## 🚀 Future Work

Future Work:

* Use LLM-based decision making instead of fixed assignment
* Optimize routes dynamically using RL agents

---

## 👥 Team

* GAGAN RAO K
* NUTHAN KUMAR
* PRABHAKARA R

---

## ✅ Status

✔️ Completed and working
