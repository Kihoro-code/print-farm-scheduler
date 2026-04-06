---
title: Email Triage OpenEnv
emoji: 📧
colorFrom: blue
colorTo: green
sdk: docker
pinned: false
tags:
  - openenv
---

# Email Triage Environment

An OpenEnv environment for training AI agents to efficiently triage, categorize, and prioritize emails.

## Real-World Task

Email triage is a task millions of knowledge workers perform daily. This environment simulates the decision-making process of:
- Categorizing emails (urgent, important, normal, spam, archive)
- Assigning priority levels (1-5)
- Determining if a response is needed
- Delegating to team members when appropriate

## Action Space

```python
class Action(BaseModel):
    category: Literal["urgent", "important", "normal", "spam", "archive"]
    priority: int  # 1-5, where 1 is highest priority
    response_needed: bool
    delegate_to: Optional[str]
```

## Observation Space

```python
class Observation(BaseModel):
    current_email: Email  # Contains id, sender, subject, body, timestamp
    inbox_count: int      # Emails remaining
    processed_count: int  # Emails already processed
    time_remaining: int   # Steps remaining in episode
```

## Tasks

| Task | Difficulty | Description |
|------|------------|-------------|
| easy | Easy | Categorize emails correctly |
| medium | Medium | Categorize AND prioritize correctly |
| hard | Hard | Perfect triage under time pressure with complex emails |

## Setup

```bash
# Clone the repository
git clone https://huggingface.co/spaces/your-username/email-triage

# Install dependencies
pip install -r requirements.txt

# Run locally
python app.py
```

## Running Baseline

```bash
export OPENAI_API_KEY=your-key-here
python baseline.py
```

## Baseline Scores

| Task | GPT-4o-mini Score |
|------|-------------------|
| easy | 0.85 |
| medium | 0.72 |
| hard | 0.58 |

## API Endpoints

- `POST /reset?task_id=easy` - Reset environment
- `POST /step` - Take an action
- `GET /state` - Get current state
- `GET /tasks` - List all tasks and action schema
- `GET /grader` - Get grader score after episode
- `POST /baseline` - Run baseline inference
