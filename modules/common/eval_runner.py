import json
import time
from typing import Any, Dict, List
from langchain_core.messages import HumanMessage, AIMessage, SystemMessage
from modules.common.agent_tracer import AgentTracer

try:
    from IPython.display import display, HTML
except ImportError:
    def display(x):
        print(x)
    def HTML(x):
        return x

def _convert_messages(raw_msgs: List[Dict[str, str]]) -> List[Any]:
    msgs = []
    for m in raw_msgs:
        role = m.get("role", "user")
        content = m.get("content", "")
        if role in ("user", "human"):
            msgs.append(HumanMessage(content=content))
        elif role in ("assistant", "ai"):
            msgs.append(AIMessage(content=content))
        elif role == "system":
            msgs.append(SystemMessage(content=content))
    return msgs

class EvalRunner:
    """
    Automated evaluation runner for AI agents.
    Executes JSON-defined test scenarios, validates them using AgentTracer outputs,
    and displays structured HTML test reports.
    """
    def __init__(self, agent: Any, tracer: AgentTracer):
        self.agent = agent
        self.tracer = tracer

    def run_scenarios(self, scenario_file_path: str) -> List[Dict[str, Any]]:
        """
        Loads and executes test scenarios from a JSON file.
        Returns a list of scenario test results.
        """
        import os
        if not os.path.exists(scenario_file_path):
            # Try searching up parent directories to handle notebooks executing from subfolders
            for parent in ("..", os.path.join("..", ".."), os.path.join("..", "..", "..")):
                alt_path = os.path.join(parent, scenario_file_path)
                if os.path.exists(alt_path):
                    scenario_file_path = alt_path
                    break

        with open(scenario_file_path, "r", encoding="utf-8") as f:
            scenarios = json.load(f)

        results = []
        for sc in scenarios:
            sc_id = sc.get("id", "scenario_unknown")
            sc_name = sc.get("name", "Unnamed Scenario")
            raw_input_messages = sc.get("input_messages", [])
            expected_behavior = sc.get("expected_behavior", "")
            criteria = sc.get("success_criteria", {})

            # Prepare configuration with thread_id for tracer checkpointing
            config = {"configurable": {"thread_id": sc_id}}
            input_msgs = _convert_messages(raw_input_messages)

            # Enable logging context on runtime
            if hasattr(self.agent, "context"):
                self.agent.context.logging_enabled = True

            print(f"🧪 Running scenario: [{sc_id}] {sc_name}...")
            
            error = None
            try:
                # Invoke agent
                # Note: create_agent returns a runnable which takes state dictionary or messages list
                self.agent.invoke({"messages": input_msgs}, config=config)
            except Exception as e:
                error = str(e)
                print(f"❌ Error during execution: {error}")

            # Retrieve logs from tracer
            run_info = self.tracer._active_runs.get(sc_id, {})
            events = run_info.get("events", [])
            final_response = run_info.get("final_response", "")

            # Perform evaluation
            failures = []
            
            # 1. Verify expected tool calls
            expected_tools = criteria.get("tool_calls", [])
            actual_tools = [ev["tool_name"] for ev in events if ev.get("type") == "tool_call"]
            
            for tool in expected_tools:
                if tool not in actual_tools:
                    failures.append(f"Missing expected tool call: '{tool}' (called: {actual_tools})")

            # 2. Verify response keywords
            expected_keywords = criteria.get("final_answer_contains", [])
            if expected_keywords:
                matched_keywords = [kw for kw in expected_keywords if kw.lower() in final_response.lower()]
                if not matched_keywords:
                    failures.append(f"Response missing expected keywords (expected at least one of): {expected_keywords}")

            success = len(failures) == 0 and error is None
            
            results.append({
                "id": sc_id,
                "name": sc_name,
                "success": success,
                "expected": expected_behavior,
                "final_response": final_response,
                "failures": failures,
                "error": error,
                "latency_ms": run_info.get("total_latency_ms", 0),
                "input_tokens": run_info.get("total_input_tokens", 0),
                "output_tokens": run_info.get("total_output_tokens", 0),
                "tool_calls": len(actual_tools),
            })
            
        return results

    def show_report(self, results: List[Dict[str, Any]]):
        """
        Renders a detailed premium dark mode HTML report summarizing the evaluation results.
        """
        total = len(results)
        passed = sum(1 for r in results if r["success"])
        failed = total - passed
        pass_rate = (passed / total * 100) if total > 0 else 0
        avg_latency = sum(r["latency_ms"] for r in results) / total if total > 0 else 0
        total_tokens = sum(r["input_tokens"] + r["output_tokens"] for r in results)

        summary_color = "#34d399" if pass_rate == 100 else ("#f59e0b" if pass_rate > 50 else "#ef4444")

        # Compile rows
        rows_html = ""
        for r in results:
            badge_color = "#34d399" if r["success"] else "#ef4444"
            status_text = "PASS" if r["success"] else "FAIL"
            
            details = ""
            if not r["success"]:
                if r["error"]:
                    details = f"<div style='color: #fca5a5; font-size: 0.75rem; margin-top: 4px;'><strong>Runtime Error:</strong> {r['error']}</div>"
                else:
                    details = "<ul style='margin: 4px 0 0 0; padding-left: 16px; color: #fca5a5; font-size: 0.75rem;'>"
                    for f in r["failures"]:
                        details += f"<li>{f}</li>"
                    details += "</ul>"
            else:
                details = f"<div style='color: #6ee7b7; font-size: 0.75rem; margin-top: 4px;'>Passed all success criteria.</div>"

            rows_html += f"""
            <tr style="border-bottom: 1px solid #334155; vertical-align: top;">
                <td style="padding: 12px; text-align: left; font-size: 0.85rem; font-weight: 700; color: #f1f5f9;">
                    {r['name']}
                    <div style="font-size: 0.7rem; color: #94a3b8; font-weight: normal; margin-top: 2px;">ID: {r['id']}</div>
                </td>
                <td style="padding: 12px;">
                    <span style="background-color: {badge_color}22; color: {badge_color}; border: 1px solid {badge_color}; padding: 2px 8px; border-radius: 4px; font-size: 0.75rem; font-weight: 700;">
                        {status_text}
                    </span>
                </td>
                <td style="padding: 12px; text-align: left; font-size: 0.75rem; color: #cbd5e1; max-width: 250px;">
                    <div style="font-style: italic; color: #94a3b8; margin-bottom: 4px;">Expected: {r['expected']}</div>
                    {details}
                </td>
                <td style="padding: 12px; font-size: 0.8rem; color: #cbd5e1;">{r['latency_ms']:,} ms</td>
                <td style="padding: 12px; font-size: 0.8rem; color: #cbd5e1;">
                    In: {r['input_tokens']}<br>Out: {r['output_tokens']}
                </td>
                <td style="padding: 12px; font-size: 0.8rem; color: #cbd5e1;">{r['tool_calls']}</td>
            </tr>
            """

        html = f"""
        <div style="background-color: #0f172a; color: #e2e8f0; font-family: 'Segoe UI', sans-serif; padding: 24px; border-radius: 16px; border: 1px solid #1e293b; max-width: 950px; margin: 20px auto; box-shadow: 0 10px 15px -3px rgba(0, 0, 0, 0.3);">
            <!-- Title -->
            <h3 style="margin: 0 0 20px 0; color: #f8fafc; font-size: 1.25rem; font-weight: 700;">🔬 Automated Evaluation Report</h3>
            
            <!-- Summary metrics -->
            <div style="display: grid; grid-template-columns: repeat(4, 1fr); gap: 12px; margin-bottom: 24px;">
                <div style="background-color: #1e293b; padding: 12px; border-radius: 8px; text-align: center; border: 1px solid #334155;">
                    <div style="font-size: 0.7rem; color: #94a3b8; text-transform: uppercase; margin-bottom: 4px;">Success Rate</div>
                    <div style="font-size: 1.3rem; font-weight: 700; color: {summary_color};">{pass_rate:.1f}%</div>
                </div>
                <div style="font-size: 0.7rem; text-align: center; background-color: #1e293b; padding: 12px; border-radius: 8px; border: 1px solid #334155;">
                    <div style="color: #94a3b8; text-transform: uppercase; margin-bottom: 4px;">Passed / Total</div>
                    <div style="font-size: 1.3rem; font-weight: 700; color: #f8fafc;">{passed} / {total}</div>
                </div>
                <div style="background-color: #1e293b; padding: 12px; border-radius: 8px; text-align: center; border: 1px solid #334155;">
                    <div style="font-size: 0.7rem; color: #94a3b8; text-transform: uppercase; margin-bottom: 4px;">Avg Latency</div>
                    <div style="font-size: 1.3rem; font-weight: 700; color: #38bdf8;">{avg_latency:.1f} ms</div>
                </div>
                <div style="background-color: #1e293b; padding: 12px; border-radius: 8px; text-align: center; border: 1px solid #334155;">
                    <div style="color: #94a3b8; text-transform: uppercase; margin-bottom: 4px;">Total Tokens</div>
                    <div style="font-size: 1.3rem; font-weight: 700; color: #a78bfa;">{total_tokens:,}</div>
                </div>
            </div>

            <!-- Details Table -->
            <table style="width: 100%; border-collapse: collapse; text-align: center; font-size: 0.9rem;">
                <thead>
                    <tr style="background-color: #1e293b; border-bottom: 2px solid #334155; color: #94a3b8; font-size: 0.75rem; text-transform: uppercase; font-weight: 700;">
                        <th style="padding: 10px; text-align: left;">Scenario Name</th>
                        <th style="padding: 10px;">Status</th>
                        <th style="padding: 10px; text-align: left;">Expected Behavior & Criteria</th>
                        <th style="padding: 10px;">Latency</th>
                        <th style="padding: 10px;">Tokens</th>
                        <th style="padding: 10px;">Tools</th>
                    </tr>
                </thead>
                <tbody>
                    {rows_html}
                </tbody>
            </table>
        </div>
        """
        display(HTML(html))
