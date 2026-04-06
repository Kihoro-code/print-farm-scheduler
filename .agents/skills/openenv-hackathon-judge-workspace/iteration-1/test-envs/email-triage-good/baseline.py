"""
Baseline inference script for Email Triage Environment
Uses OpenAI API to run a model against all tasks
"""

import os
import json
from openai import OpenAI
from env import EmailTriageEnv, Action, TASKS


def run_baseline():
    # Read API key from environment
    api_key = os.environ.get("OPENAI_API_KEY")
    if not api_key:
        raise ValueError("OPENAI_API_KEY environment variable not set")

    client = OpenAI(api_key=api_key)

    results = {}

    for task_id, task_config in TASKS.items():
        print(f"\n=== Running task: {task_id} ===")

        env = EmailTriageEnv(task_id=task_id)
        obs = env.reset()
        done = False

        while not done:
            # Format observation for LLM
            prompt = f"""You are triaging emails. Current email:
            
From: {obs.current_email.sender}
Subject: {obs.current_email.subject}
Body: {obs.current_email.body}

Emails remaining: {obs.inbox_count}
Time remaining: {obs.time_remaining} steps

Respond with JSON containing:
- category: one of "urgent", "important", "normal", "spam", "archive"
- priority: 1-5 (1=highest)
- response_needed: true/false
- delegate_to: null or team member name
"""

            response = client.chat.completions.create(
                model="gpt-4o-mini",
                messages=[{"role": "user", "content": prompt}],
                response_format={"type": "json_object"},
            )

            action_data = json.loads(response.choices[0].message.content)
            action = Action(**action_data)

            obs, reward, done, info = env.step(action)
            print(f"  Step {info['processed']}: score={reward.score:.2f}")

        # Grade the task
        grader = task_config["grader"]
        final_score = grader(env)
        results[task_id] = final_score
        print(f"Task {task_id} final score: {final_score:.3f}")

    print("\n=== BASELINE RESULTS ===")
    for task_id, score in results.items():
        print(f"{task_id}: {score:.3f}")

    return results


if __name__ == "__main__":
    run_baseline()
