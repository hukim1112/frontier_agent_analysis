"""
app/utils/log_analyzer.py
=========================
에이전트 감사 로그(Audit Trail) 및 실행 궤적(Observability) 분석기.
artifacts/logs 디렉토리의 JSONL 로그를 파싱하여 세션 수, 호출 횟수, 레이턴시,
도구별 사용 빈도 등의 핵심 운영 메트릭을 요약 분석하고 시각화합니다.

실행 방법:
  python -m app.utils.log_analyzer
"""

import os
import sys
import json
import glob
from typing import Dict, Any, List
from collections import Counter


def analyze_logs(log_dir: str = "./artifacts/logs") -> Dict[str, Any]:
    """
    지정된 디렉토리의 모든 로그 파일(.json, .jsonl)을 스캔하여
    에이전트 실행 통계를 집계합니다.
    """
    stats: Dict[str, Any] = {
        "total_sessions": 0,
        "total_executions": 0,
        "total_tool_calls": 0,
        "agent_latencies": [],
        "tool_latencies": [],
        "tool_usage": Counter(),
        "tool_latency_by_name": {},
        "sessions": set(),
        "log_files_count": 0,
    }

    if not os.path.exists(log_dir):
        return stats

    log_files = (
        glob.glob(os.path.join(log_dir, "*.jsonl"))
        + glob.glob(os.path.join(log_dir, "*.json"))
        + glob.glob(os.path.join(log_dir, "**", "*.jsonl"), recursive=True)
        + glob.glob(os.path.join(log_dir, "**", "*.json"), recursive=True)
    )
    log_files = sorted(list(dict.fromkeys(log_files)))
    stats["log_files_count"] = len(log_files)

    for fpath in log_files:
        try:
            with open(fpath, "r", encoding="utf-8") as f:
                for line in f:
                    line = line.strip()
                    if not line:
                        continue
                    try:
                        record = json.loads(line)
                    except Exception:
                        continue

                    # session_id 집계
                    sid = record.get("session_id") or record.get("thread_id") or "unknown"
                    stats["sessions"].add(sid)

                    event_type = record.get("event") or record.get("type") or ""

                    # 에이전트 실행 이벤트
                    if event_type in ["agent_execution", "run_summary", "agent_call"]:
                        stats["total_executions"] += 1
                        lat = record.get("latency_ms") or record.get("duration_ms")
                        if lat is not None:
                            stats["agent_latencies"].append(float(lat))

                    # 도구 실행 이벤트
                    elif event_type in ["tool_execution", "tool_call"]:
                        stats["total_tool_calls"] += 1
                        tname = record.get("tool_name") or record.get("name") or "unknown_tool"
                        stats["tool_usage"][tname] += 1

                        lat = record.get("latency_ms") or record.get("duration_ms")
                        if lat is not None:
                            stats["tool_latencies"].append(float(lat))
                            if tname not in stats["tool_latency_by_name"]:
                                stats["tool_latency_by_name"][tname] = []
                            stats["tool_latency_by_name"][tname].append(float(lat))

                    # 일반 AgentLogTracer 세션 트리 레코드
                    elif "tool_calls_count" in record or "total_latency_ms" in record:
                        stats["total_executions"] += 1
                        stats["total_tool_calls"] += record.get("tool_calls_count", 0)
                        lat = record.get("total_latency_ms")
                        if lat:
                            stats["agent_latencies"].append(float(lat))
                        for t in record.get("tools_used", []):
                            stats["tool_usage"][t] += 1

        except Exception as e:
            print(f"⚠️ 로그 파일 읽기 실패 ({fpath}): {e}")

    stats["total_sessions"] = len(stats["sessions"])
    return stats


def print_log_dashboard(stats: Dict[str, Any]):
    """집계된 통계 데이터를 터미널 대시보드로 시각화 출력합니다."""
    print("\n" + "=" * 70)
    print("📊 [Agent Observability] 세션 감사 로그 분석 대시보드")
    print("=" * 70)

    total_sessions = stats["total_sessions"]
    total_exec = stats["total_executions"]
    total_tools = stats["total_tool_calls"]

    avg_agent_lat = (
        sum(stats["agent_latencies"]) / len(stats["agent_latencies"])
        if stats["agent_latencies"]
        else 0.0
    )
    avg_tool_lat = (
        sum(stats["tool_latencies"]) / len(stats["tool_latencies"])
        if stats["tool_latencies"]
        else 0.0
    )

    print(f"  • 분석 대상 로그 파일  : {stats['log_files_count']} 개")
    print(f"  • 고유 세션 수 (Sessions): {total_sessions} 개")
    print(f"  • 총 에이전트 실행 횟수: {total_exec} 회")
    print(f"  • 총 도구(Tool) 격발 수: {total_tools} 회")
    print(f"  • 평균 에이전트 레이턴시: {avg_agent_lat:.1f} ms")
    print(f"  • 평균 도구 실행 시간  : {avg_tool_lat:.1f} ms")

    print("\n[🛠️ 도구별 사용 빈도 및 평균 레이턴시 TOP 5]")
    print("-" * 70)
    print(f"  {'순위':<4} | {'도구 이름':<26} | {'호출 횟수':<10} | {'평균 레이턴시'}")
    print("-" * 70)

    top_tools = stats["tool_usage"].most_common(5)
    if not top_tools:
        print("  (기록된 도구 실행 로그가 없습니다)")
    else:
        for rank, (tname, count) in enumerate(top_tools, start=1):
            t_lats = stats["tool_latency_by_name"].get(tname, [])
            avg_t_lat = f"{sum(t_lats)/len(t_lats):.1f} ms" if t_lats else "N/A"
            print(f"  #{rank:<3} | {tname:<26} | {count:<10} | {avg_t_lat}")

    print("=" * 70 + "\n")


if __name__ == "__main__":
    target_dir = sys.argv[1] if len(sys.argv) > 1 else "./artifacts/logs"
    stats_data = analyze_logs(target_dir)
    print_log_dashboard(stats_data)
