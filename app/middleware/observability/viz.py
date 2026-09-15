import json
from typing import Any, Dict

def render_timeline_html(run_info: Dict[str, Any]) -> str:
    """
    Renders a premium dark-themed HTML timeline for an agent run session.
    """
    session_id = run_info.get("session_id", "unknown")
    user_query = run_info.get("user_query", "N/A")
    status = run_info.get("status", "SUCCESS")
    total_latency = run_info.get("total_latency_ms", 0)
    total_in_tokens = run_info.get("total_input_tokens", 0)
    total_out_tokens = run_info.get("total_output_tokens", 0)
    total_calls = run_info.get("total_tool_calls", 0)
    events = run_info.get("events", [])

    status_color = "#34d399" if status == "SUCCESS" else "#f87171"
    
    html = f"""
    <div style="background-color: #0f172a; color: #e2e8f0; font-family: 'Segoe UI', -apple-system, BlinkMacSystemFont, Roboto, sans-serif; padding: 24px; border-radius: 16px; box-shadow: 0 10px 15px -3px rgba(0, 0, 0, 0.3), 0 4px 6px -2px rgba(0, 0, 0, 0.05); border: 1px solid #1e293b; max-width: 900px; margin: 20px auto;">
        <!-- Header Section -->
        <div style="border-bottom: 1px solid #1e293b; padding-bottom: 16px; margin-bottom: 20px;">
            <div style="display: flex; justify-content: space-between; align-items: center;">
                <h3 style="margin: 0; color: #f8fafc; font-size: 1.25rem; font-weight: 700;">🔄 Agent Session Timeline</h3>
                <span style="background-color: {status_color}22; color: {status_color}; border: 1px solid {status_color}; padding: 4px 12px; border-radius: 9999px; font-size: 0.75rem; font-weight: 700; text-transform: uppercase;">
                    {status}
                </span>
            </div>
            <div style="margin-top: 4px; font-size: 0.8rem; color: #64748b;">Session ID: <code style="color: #cbd5e1; font-family: monospace;">{session_id}</code></div>
        </div>

        <!-- Query Card -->
        <div style="background-color: #1e293b; border-radius: 8px; padding: 16px; margin-bottom: 24px; border-left: 4px solid #38bdf8;">
            <div style="font-size: 0.75rem; text-transform: uppercase; color: #94a3b8; font-weight: 700; letter-spacing: 0.05em; margin-bottom: 4px;">User Query</div>
            <div style="font-size: 0.95rem; color: #f1f5f9; line-height: 1.5; white-space: pre-wrap;">{user_query}</div>
        </div>

        <!-- Stats Overview -->
        <div style="display: grid; grid-template-columns: repeat(4, 1fr); gap: 12px; margin-bottom: 30px;">
            <div style="background-color: #1e293b; padding: 12px; border-radius: 8px; text-align: center; border: 1px solid #334155;">
                <div style="font-size: 0.7rem; color: #94a3b8; text-transform: uppercase; margin-bottom: 4px;">Total Latency</div>
                <div style="font-size: 1.15rem; font-weight: 700; color: #38bdf8;">{total_latency:,} ms</div>
            </div>
            <div style="background-color: #1e293b; padding: 12px; border-radius: 8px; text-align: center; border: 1px solid #334155;">
                <div style="font-size: 0.7rem; color: #94a3b8; text-transform: uppercase; margin-bottom: 4px;">Input Tokens</div>
                <div style="font-size: 1.15rem; font-weight: 700; color: #a78bfa;">{total_in_tokens:,}</div>
            </div>
            <div style="background-color: #1e293b; padding: 12px; border-radius: 8px; text-align: center; border: 1px solid #334155;">
                <div style="font-size: 0.7rem; color: #94a3b8; text-transform: uppercase; margin-bottom: 4px;">Output Tokens</div>
                <div style="font-size: 1.15rem; font-weight: 700; color: #f472b6;">{total_out_tokens:,}</div>
            </div>
            <div style="background-color: #1e293b; padding: 12px; border-radius: 8px; text-align: center; border: 1px solid #334155;">
                <div style="font-size: 0.7rem; color: #94a3b8; text-transform: uppercase; margin-bottom: 4px;">Tool Calls</div>
                <div style="font-size: 1.15rem; font-weight: 700; color: #34d399;">{total_calls}</div>
            </div>
        </div>

        <!-- Event Timeline -->
        <div style="position: relative; border-left: 2px solid #334155; margin-left: 16px; padding-left: 24px; padding-top: 8px;">
    """

    for idx, ev in enumerate(events):
        ev_type = ev.get("type", "unknown")
        timestamp = ev.get("timestamp", 0)
        latency = ev.get("latency_ms", 0)

        if ev_type == "model_call":
            in_tokens = ev.get("input_tokens", 0)
            out_tokens = ev.get("output_tokens", 0)
            msg_count = ev.get("input_messages_count", 0)
            response_text = ev.get("response_text", "")
            
            # Extract main thoughts or brief version of the answer
            preview_text = response_text if len(response_text) < 300 else response_text[:300] + "..."
            
            html += f"""
            <div style="position: relative; margin-bottom: 24px;">
                <!-- Timeline bullet -->
                <div style="position: absolute; left: -33px; top: 2px; width: 16px; height: 16px; border-radius: 9999px; background-color: #a78bfa; border: 4px solid #0f172a; box-shadow: 0 0 0 2px #a78bfa55;"></div>
                
                <!-- Card -->
                <div style="background-color: #1e293b; border-radius: 8px; padding: 14px; border: 1px solid #4c1d95;">
                    <div style="display: flex; justify-content: space-between; margin-bottom: 8px; align-items: center;">
                        <span style="font-weight: 700; color: #c084fc; font-size: 0.85rem; display: flex; align-items: center; gap: 6px;">
                            🧠 LLM Inference Step {idx + 1}
                        </span>
                        <span style="font-size: 0.75rem; color: #94a3b8; background-color: #0f172a; padding: 2px 8px; border-radius: 4px;">
                            ⚡ {latency:,} ms
                        </span>
                    </div>
                    
                    <div style="font-size: 0.8rem; color: #cbd5e1; margin-bottom: 8px; line-height: 1.4; white-space: pre-wrap;">{preview_text}</div>
                    
                    <div style="display: flex; gap: 12px; font-size: 0.7rem; color: #94a3b8; border-top: 1px solid #334155; padding-top: 8px;">
                        <span>📥 Messages: <strong>{msg_count}</strong></span>
                        <span>🏷️ Input Tokens: <strong style="color: #c084fc;">{in_tokens}</strong></span>
                        <span>🏷️ Output Tokens: <strong style="color: #f472b6;">{out_tokens}</strong></span>
                    </div>
                </div>
            </div>
            """
        elif ev_type == "tool_call":
            tool_name = ev.get("tool_name", "unknown")
            tool_args = ev.get("arguments", {})
            result = ev.get("result", "")
            t_status = ev.get("status", "SUCCESS")
            
            tool_status_color = "#34d399" if t_status == "SUCCESS" else "#ef4444"
            preview_result = result if len(result) < 300 else result[:300] + "..."
            
            html += f"""
            <div style="position: relative; margin-bottom: 24px;">
                <!-- Timeline bullet -->
                <div style="position: absolute; left: -33px; top: 2px; width: 16px; height: 16px; border-radius: 9999px; background-color: #34d399; border: 4px solid #0f172a; box-shadow: 0 0 0 2px #34d39955;"></div>
                
                <!-- Card -->
                <div style="background-color: #1e293b; border-radius: 8px; padding: 14px; border: 1px solid #064e3b;">
                    <div style="display: flex; justify-content: space-between; margin-bottom: 8px; align-items: center;">
                        <span style="font-weight: 700; color: #34d399; font-size: 0.85rem; display: flex; align-items: center; gap: 6px;">
                            🔧 Tool Call: <code style="color: #6ee7b7; font-family: monospace; background-color: #0f172a; padding: 2px 6px; border-radius: 4px;">{tool_name}</code>
                        </span>
                        <div style="display: flex; gap: 8px; align-items: center;">
                            <span style="font-size: 0.7rem; color: {tool_status_color}; background-color: {tool_status_color}15; padding: 2px 6px; border-radius: 4px; border: 1px solid {tool_status_color}30;">
                                {t_status}
                            </span>
                            <span style="font-size: 0.75rem; color: #94a3b8; background-color: #0f172a; padding: 2px 8px; border-radius: 4px;">
                                ⚡ {latency:,} ms
                            </span>
                        </div>
                    </div>
                    
                    <div style="margin-bottom: 8px;">
                        <span style="font-size: 0.7rem; color: #94a3b8; font-weight: 700; display: block; margin-bottom: 2px;">ARGUMENTS</span>
                        <pre style="margin: 0; background-color: #0f172a; padding: 8px; border-radius: 4px; font-family: monospace; font-size: 0.75rem; color: #f1f5f9; overflow-x: auto;">{json.dumps(tool_args, indent=2, ensure_ascii=False)}</pre>
                    </div>

                    <div>
                        <span style="font-size: 0.7rem; color: #94a3b8; font-weight: 700; display: block; margin-bottom: 2px;">RESULT</span>
                        <div style="font-size: 0.75rem; color: #cbd5e1; background-color: #0f172a; padding: 8px; border-radius: 4px; font-family: monospace; max-height: 120px; overflow-y: auto; white-space: pre-wrap; line-height: 1.4;">{preview_result}</div>
                    </div>
                </div>
            </div>
            """
    
    # End of timeline wrapper and final agent response
    final_resp = run_info.get("final_response", "No response")
    html += f"""
        </div> <!-- Timeline wrapper end -->

        <!-- Final Response Card -->
        <div style="margin-top: 16px; background-color: #1e293b; border-radius: 8px; padding: 16px; border: 1px solid #334155;">
            <div style="font-size: 0.75rem; text-transform: uppercase; color: #38bdf8; font-weight: 700; letter-spacing: 0.05em; margin-bottom: 6px; display: flex; align-items: center; gap: 6px;">
                🏁 Final Agent Response
            </div>
            <div style="font-size: 0.95rem; color: #f8fafc; line-height: 1.6; white-space: pre-wrap;">{final_resp}</div>
        </div>
    </div>
    """
    return html


def render_tool_summary_html(tool_stats: Dict[str, Dict[str, Any]]) -> str:
    """
    Renders an HTML table summarizing the tool usages, success rates and latency.
    """
    rows_html = ""
    for tool_name, stats in tool_stats.items():
        calls = stats.get("calls", 0)
        failures = stats.get("failures", 0)
        successes = calls - failures
        success_rate = (successes / calls * 100) if calls > 0 else 0
        avg_latency = stats.get("total_latency", 0) / calls if calls > 0 else 0
        
        rate_color = "#34d399" if success_rate == 100 else ("#f59e0b" if success_rate > 0 else "#ef4444")
        
        rows_html += f"""
        <tr style="border-bottom: 1px solid #334155;">
            <td style="padding: 12px; font-family: monospace; color: #6ee7b7; text-align: left; font-size: 0.85rem;">{tool_name}</td>
            <td style="padding: 12px; font-weight: bold; color: #e2e8f0;">{calls}</td>
            <td style="padding: 12px; color: #34d399;">{successes}</td>
            <td style="padding: 12px; color: #ef4444;">{failures}</td>
            <td style="padding: 12px; color: {rate_color}; font-weight: 700;">{success_rate:.1f}%</td>
            <td style="padding: 12px; color: #38bdf8;">{avg_latency:.1f} ms</td>
        </tr>
        """
        
    if not tool_stats:
        rows_html = """
        <tr>
            <td colspan="6" style="padding: 20px; color: #94a3b8; font-style: italic;">No tools were executed.</td>
        </tr>
        """

    html = f"""
    <div style="background-color: #0f172a; color: #e2e8f0; font-family: 'Segoe UI', sans-serif; padding: 24px; border-radius: 16px; border: 1px solid #1e293b; max-width: 900px; margin: 20px auto; box-shadow: 0 10px 15px -3px rgba(0, 0, 0, 0.3);">
        <h3 style="margin: 0 0 16px 0; color: #f8fafc; font-size: 1.15rem; font-weight: 700; display: flex; align-items: center; gap: 8px;">
            📊 Tool Execution Statistics
        </h3>
        <table style="width: 100%; border-collapse: collapse; text-align: center; font-size: 0.9rem;">
            <thead>
                <tr style="background-color: #1e293b; border-bottom: 2px solid #334155; color: #94a3b8; font-size: 0.75rem; text-transform: uppercase; font-weight: 700;">
                    <th style="padding: 10px; text-align: left;">Tool Name</th>
                    <th style="padding: 10px;">Total Calls</th>
                    <th style="padding: 10px; color: #34d399;">Success</th>
                    <th style="padding: 10px; color: #ef4444;">Failure</th>
                    <th style="padding: 10px;">Success Rate</th>
                    <th style="padding: 10px;">Avg Latency</th>
                </tr>
            </thead>
            <tbody>
                {rows_html}
            </tbody>
        </table>
    </div>
    """
    return html


def render_loop_architecture_svg() -> str:
    """
    Returns a complex, beautiful, premium dark mode SVG showcasing the 4-phase Agent Loop.
    """
    return """
<div align="center">
<svg width="800" height="520" viewBox="0 0 800 520" fill="none" xmlns="http://www.w3.org/2000/svg">
  <rect width="800" height="520" rx="14" fill="#0F172A"/>
  <rect x="1" y="1" width="798" height="518" rx="13" stroke="#1E293B" stroke-width="2"/>
  
  <!-- Title -->
  <text x="400" y="35" fill="#F8FAFC" font-size="18" font-family="'Segoe UI', -apple-system, sans-serif" font-weight="bold" text-anchor="middle">🔄 Agent Loop Architecture: 4-Phase Core Cycle</text>
  <text x="400" y="55" fill="#64748B" font-size="11" font-family="'Segoe UI', sans-serif" text-anchor="middle">A standard execution pipeline for stateful AI agents</text>

  <!-- Phase 1: Prompt Assembly -->
  <g transform="translate(40, 90)">
    <rect x="0" y="0" width="220" height="140" rx="10" fill="#1E293B" stroke="#38BDF8" stroke-width="1.5"/>
    <text x="12" y="24" fill="#38BDF8" font-size="12" font-family="Segoe UI" font-weight="bold">📋 Phase 1: Prompt Assembly</text>
    
    <rect x="10" y="38" width="200" height="22" rx="4" fill="#0F172A" stroke="#334155"/>
    <text x="18" y="52" fill="#A5B4FC" font-size="9" font-family="monospace">System Instructions (Static)</text>
    
    <rect x="10" y="64" width="200" height="22" rx="4" fill="#0F172A" stroke="#334155"/>
    <text x="18" y="78" fill="#FCD34D" font-size="9" font-family="monospace">Memory &amp; Conversation History</text>
    
    <rect x="10" y="90" width="200" height="22" rx="4" fill="#0F172A" stroke="#334155"/>
    <text x="18" y="104" fill="#34D399" font-size="9" font-family="monospace">Tool Schemas &amp; API specs</text>
    
    <rect x="10" y="116" width="200" height="18" rx="4" fill="#0F172A" stroke="#F97316"/>
    <text x="18" y="128" fill="#F97316" font-size="8" font-family="monospace">User Query / Error Feedbacks</text>
  </g>

  <!-- Connection: P1 -> P2 -->
  <path d="M 265 160 L 305 160" stroke="#38BDF8" stroke-width="2" marker-end="url(#arrowBlue)"/>
  <text x="285" y="152" fill="#64748B" font-size="8" font-family="monospace" text-anchor="middle">Tokens</text>

  <!-- Phase 2: LLM Inference -->
  <g transform="translate(310, 110)">
    <rect x="0" y="0" width="180" height="100" rx="10" fill="#1E293B" stroke="#A78BFA" stroke-width="1.5"/>
    <text x="12" y="24" fill="#A78BFA" font-size="12" font-family="Segoe UI" font-weight="bold">🧠 Phase 2: LLM Inference</text>
    
    <rect x="10" y="38" width="160" height="28" rx="4" fill="#0F172A" stroke="#334155"/>
    <text x="18" y="55" fill="#E2E8F0" font-size="9" font-family="monospace">Generate AIMessage</text>
    
    <rect x="10" y="70" width="75" height="20" rx="4" fill="#7C3AED"/>
    <text x="16" y="83" fill="#F8FAFC" font-size="8" font-family="Segoe UI" font-weight="bold">⚡ Streaming</text>
    
    <rect x="95" y="70" width="75" height="20" rx="4" fill="#0369A1"/>
    <text x="101" y="83" fill="#38BDF8" font-size="8" font-family="Segoe UI" font-weight="bold">97% Cache Hit</text>
  </g>

  <!-- Connection: P2 -> P3 -->
  <path d="M 495 160 L 535 160" stroke="#A78BFA" stroke-width="2" marker-end="url(#arrowPurple)"/>
  <text x="515" y="152" fill="#64748B" font-size="8" font-family="monospace" text-anchor="middle">ToolCalls</text>

  <!-- Phase 3: Tool Execution -->
  <g transform="translate(540, 90)">
    <rect x="0" y="0" width="220" height="140" rx="10" fill="#1E293B" stroke="#34D399" stroke-width="1.5"/>
    <text x="12" y="24" fill="#34D399" font-size="12" font-family="Segoe UI" font-weight="bold">🔧 Phase 3: Tool Execution</text>
    
    <rect x="10" y="38" width="200" height="26" rx="4" fill="#0F172A" stroke="#334155"/>
    <text x="18" y="54" fill="#E2E8F0" font-size="9" font-family="monospace">Execute Tool Schemas</text>
    
    <rect x="10" y="70" width="95" height="30" rx="4" fill="#065F46" stroke="#34D399" stroke-dasharray="3"/>
    <text x="16" y="84" fill="#34D399" font-size="8" font-family="Segoe UI">✅ Read Tools</text>
    <text x="16" y="94" fill="#6EE7B7" font-size="7" font-family="Segoe UI">Safe, Parallelized</text>
    
    <rect x="115" y="70" width="95" height="30" rx="4" fill="#7F1D1D" stroke="#EF4444" stroke-dasharray="3"/>
    <text x="121" y="84" fill="#EF4444" font-size="8" font-family="Segoe UI">🔒 Write/Exec Tools</text>
    <text x="121" y="94" fill="#FCA5A5" font-size="7" font-family="Segoe UI">Unsafe, Mutex-Locked</text>

    <rect x="10" y="106" width="200" height="24" rx="4" fill="#0F172A" stroke="#F97316"/>
    <text x="16" y="120" fill="#F97316" font-size="8" font-family="Segoe UI" font-weight="bold">🛡️ HITL Permission Gate check</text>
  </g>

  <!-- Phase 4: Self-Correction & Evaluation -->
  <g transform="translate(260, 270)">
    <rect x="0" y="0" width="280" height="110" rx="10" fill="#1E293B" stroke="#FBBF24" stroke-width="1.5"/>
    <text x="12" y="24" fill="#FBBF24" font-size="12" font-family="Segoe UI" font-weight="bold">🛡️ Phase 4: Self-Correction &amp; Evaluation</text>
    
    <rect x="10" y="38" width="260" height="26" rx="4" fill="#0F172A" stroke="#334155"/>
    <text x="18" y="54" fill="#E2E8F0" font-size="9" font-family="monospace">Diagnose tool errors / verify build outputs</text>
    
    <rect x="10" y="72" width="120" height="26" rx="4" fill="#065F46"/>
    <text x="18" y="88" fill="#34D399" font-size="9" font-family="Segoe UI" font-weight="bold">✅ Pass -> Return</text>
    
    <rect x="140" y="72" width="130" height="26" rx="4" fill="#7F1D1D"/>
    <text x="148" y="88" fill="#EF4444" font-size="9" font-family="Segoe UI" font-weight="bold">❌ Fail -> Correct</text>
  </g>

  <!-- Connection: Phase 2 -> Phase 4 (Final Answer) -->
  <path d="M 400 210 L 400 265" stroke="#A78BFA" stroke-width="2" marker-end="url(#arrowPurple)"/>
  <text x="415" y="238" fill="#64748B" font-size="8" font-family="monospace">FinalAnswer</text>

  <!-- Loop back path (Phase 3 -> Phase 4: Outputs) -->
  <path d="M 650 230 L 650 325 L 545 325" stroke="#34D399" stroke-width="2" fill="none" marker-end="url(#arrowGreen)"/>
  <text x="610" y="318" fill="#64748B" font-size="8" font-family="monospace" text-anchor="middle">Outputs</text>

  <!-- Loop back path (Phase 4 Fail -> Phase 1) -->
  <path d="M 260 325 L 150 325 L 150 235" stroke="#EF4444" stroke-width="2" stroke-dasharray="6 3" fill="none" marker-end="url(#arrowRed)"/>
  <text x="100" y="300" fill="#EF4444" font-size="9" font-family="Segoe UI" font-weight="bold" text-anchor="middle">Self-Correction Feedback</text>

  <!-- Connection: Phase 4 Pass -> Finish -->
  <path d="M 400 382 L 400 440" stroke="#34D399" stroke-width="2" marker-end="url(#arrowGreen)"/>
  <text x="415" y="415" fill="#64748B" font-size="9" font-family="Segoe UI">Terminated</text>

  <!-- Finish Node -->
  <g transform="translate(320, 445)">
    <rect x="0" y="0" width="160" height="35" rx="17.5" fill="#022c22" stroke="#34D399" stroke-width="1.5"/>
    <text x="80" y="22" fill="#34D399" font-size="11" font-family="Segoe UI" font-weight="bold" text-anchor="middle">🏁 Final Answer Ready</text>
  </g>

  <!-- Arrow Markers -->
  <defs>
    <marker id="arrowBlue" viewBox="0 0 10 10" refX="6" refY="5" markerWidth="6" markerHeight="6" orient="auto">
      <path d="M0 0 L10 5 L0 10 z" fill="#38BDF8"/>
    </marker>
    <marker id="arrowPurple" viewBox="0 0 10 10" refX="6" refY="5" markerWidth="6" markerHeight="6" orient="auto">
      <path d="M0 0 L10 5 L0 10 z" fill="#A78BFA"/>
    </marker>
    <marker id="arrowGreen" viewBox="0 0 10 10" refX="6" refY="5" markerWidth="6" markerHeight="6" orient="auto">
      <path d="M0 0 L10 5 L0 10 z" fill="#34D399"/>
    </marker>
    <marker id="arrowRed" viewBox="0 0 10 10" refX="6" refY="5" markerWidth="6" markerHeight="6" orient="auto">
      <path d="M0 0 L10 5 L0 10 z" fill="#EF4444"/>
    </marker>
  </defs>
</svg>
</div>
"""
