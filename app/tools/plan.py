"""
===============================================================================
[AAWS Tool Module] Planning & Task Management Tool Engine (plan.py)
===============================================================================
Supervisor 및 오케스트레이션 에이전트를 위한 5대 Planning & Task Board 도구 엔진:
1. enter_plan: 거시적 실행 계획 수립 및 Planning 모드 진입
2. exit_plan: Planning 모드 종료 및 상태 완료 처리
3. task_create: Task Board에 세부 하위 과제 등록 (ID 자동 발급)
4. task_list: 현재 Task Board의 전체 작업 목록 및 상태 조회
5. task_update: 특정 과제 상태 갱신 (PENDING, IN_PROGRESS, COMPLETED, BLOCKED)
===============================================================================
"""

import os
import time
import json
from typing import List, Dict, Any, Optional
from langchain_core.tools import tool
from pydantic import BaseModel, Field

# =============================================================================
# Category 1: Planning Tools (enter_plan, exit_plan)
# =============================================================================
plan_session_state: Dict[str, Any] = {"active": False, "plan_name": "", "steps": []}

class EnterPlanInput(BaseModel):
    plan_name: str = Field(description="Title or concise name of the execution plan.")
    steps: List[str] = Field(description="Sequential step-by-step list of plan actions.")

@tool(args_schema=EnterPlanInput)
def enter_plan(plan_name: str, steps: List[str]) -> str:
    """Enters Planning Mode and registers a step-by-step execution plan.

    Args:
        plan_name: Title or concise name of the plan.
        steps: List of string steps detailing the sequence of actions.

    Returns:
        Status message confirming planning mode entry and step count.
    """
    global plan_session_state
    plan_session_state = {"active": True, "plan_name": plan_name, "steps": steps}
    
    # plan_state.json 파일로 저장하여 디스크 칠판 동기화
    try:
        with open("plan_state.json", "w", encoding="utf-8") as f:
            json.dump(plan_session_state, f, indent=2, ensure_ascii=False)
    except Exception:
        pass
        
    return f"[PLANNING MODE ENTERED] Registered '{plan_name}' with {len(steps)} sequential steps."


class ExitPlanInput(BaseModel):
    plan_name: str = Field(description="Title or concise name of the plan to exit.")
    status: str = Field(default="COMPLETED", description="Final status of the plan (e.g. COMPLETED, ABORTED). Defaults to COMPLETED.")

@tool(args_schema=ExitPlanInput)
def exit_plan(plan_name: str, status: str = "COMPLETED") -> str:
    """Exits Planning Mode and resumes standard execution mode.

    Args:
        plan_name: Title or name of the plan to conclude.
        status: Plan outcome status string. Defaults to 'COMPLETED'.

    Returns:
        Status message confirming plan termination.
    """
    global plan_session_state
    plan_session_state["active"] = False
    plan_session_state["status"] = status
    
    try:
        with open("plan_state.json", "w", encoding="utf-8") as f:
            json.dump(plan_session_state, f, indent=2, ensure_ascii=False)
    except Exception:
        pass
        
    return f"[PLANNING MODE EXITED] Plan '{plan_name}' terminated with status '{status}'."


# =============================================================================
# Category 2: Task Board Tools (task_create, task_list, task_update)
# =============================================================================
class TaskBoardState:
    """Manages persistent task items on the agent task board."""
    def __init__(self, filename="task_state.json"):
        self.filename = filename
        self.tasks = []
        self._load()

    def _load(self):
        if os.path.exists(self.filename):
            try:
                with open(self.filename, "r", encoding="utf-8") as f:
                    self.tasks = json.load(f)
            except Exception:
                self.tasks = []
        else:
            self.tasks = []

    def _save(self):
        try:
            with open(self.filename, "w", encoding="utf-8") as f:
                json.dump(self.tasks, f, indent=2, ensure_ascii=False)
        except Exception:
            pass

    def create_task(self, description: str) -> Dict[str, Any]:
        t_id = len(self.tasks) + 1
        t_item = {"id": t_id, "description": description, "status": "PENDING", "created_at": time.time()}
        self.tasks.append(t_item)
        self._save()
        return t_item

    def list_tasks(self) -> List[Dict[str, Any]]:
        self._load()
        return self.tasks

    def update_task(self, task_id: int, status: str) -> Optional[Dict[str, Any]]:
        self._load()
        for t in self.tasks:
            if t["id"] == task_id:
                t["status"] = status
                self._save()
                return t
        return None

task_board = TaskBoardState()

class TaskCreateInput(BaseModel):
    task_description: str = Field(description="Clear description of the sub-task item to track.")

@tool(args_schema=TaskCreateInput)
def task_create(task_description: str) -> str:
    """Creates a new tracked sub-task item on the task board.

    Args:
        task_description: Clear description of the task item.

    Returns:
        Status message with generated task ID and initial PENDING status.
    """
    t = task_board.create_task(task_description)
    return f"[TASK CREATED] Task #{t['id']}: '{task_description}' (Status: PENDING)"


@tool
def task_list() -> str:
    """Lists all active and completed tasks on the task board.

    Returns:
        JSON string listing all tracked tasks with their IDs, descriptions, and statuses.
    """
    tasks = task_board.list_tasks()
    if not tasks:
        return "No tasks currently registered on task board."
    return json.dumps(tasks, indent=2, ensure_ascii=False)


class TaskUpdateInput(BaseModel):
    task_id: int = Field(description="Numeric ID of the target task to update.")
    status: str = Field(description="New status string: 'PENDING', 'IN_PROGRESS', 'COMPLETED', or 'BLOCKED'.")

@tool(args_schema=TaskUpdateInput)
def task_update(task_id: int, status: str) -> str:
    """Updates the status of a specific task on the task board.

    Args:
        task_id: Numeric ID of the task to modify.
        status: New status string ('PENDING', 'IN_PROGRESS', 'COMPLETED', 'BLOCKED').

    Returns:
        Confirmation message of status update or error if ID not found.
    """
    t = task_board.update_task(task_id, status)
    if not t:
        return f"Task #{task_id} not found."
    return f"[TASK UPDATED] Task #{task_id} status changed to '{status}'."

# 📋 5대 Planning & Task 도구 묶음
tools_planning = [enter_plan, exit_plan, task_create, task_list, task_update]
PLANNING_AND_TASK_TOOLS = tools_planning
