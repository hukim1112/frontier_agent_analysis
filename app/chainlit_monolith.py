import os
import sys
import uuid
import re
import inspect
import chainlit as cl
from chainlit.types import ThreadDict
from typing import Dict, Any, Optional

# Setup project root
project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if project_root not in sys.path:
    sys.path.append(project_root)

import importlib
from app.utils.database import ChainlitSQLiteDataLayer, CHAINLIT_DB_PATH
from app.utils.context import AgentContext
from app.utils.message_utils import sanitize_text, normalize_content

async def _get_agent_executor(agent_name: str = "chatbot"):
    """
    지정된 에이전트 이름에 맞는 모듈을 동적으로 임포트하여 executor를 생성합니다.
    """
    try:
        module_path = f"app.agents.{agent_name}"
        if module_path in sys.modules:
            module = importlib.reload(sys.modules[module_path])
        else:
            module = importlib.import_module(module_path)
            
        factory = getattr(module, "create_agent_executor", None)
        if not factory:
            factory = getattr(module, f"create_{agent_name}_agent", None)
            if not factory:
                factory = getattr(module, "get_agent_executor", None)
                
        if not factory:
            raise AttributeError(f"Module '{module_path}' has no 'create_agent_executor' function.")
            
        if inspect.iscoroutinefunction(factory):
            return await factory()
        res = factory()
        if inspect.iscoroutine(res):
            return await res
        return res
    except Exception as e:
        print(f"❌ Failed to load agent '{agent_name}': {e}")
        # 폴백: chatbot 모듈 시도
        if agent_name != "chatbot":
            return await _get_agent_executor("chatbot")
        raise e

# 1. Chainlit Data Layer 등록 (SQLite 기반 과거 대화 목록 사이드바 및 영구 저장)
@cl.data_layer
def get_data_layer():
    return ChainlitSQLiteDataLayer(db_path=CHAINLIT_DB_PATH)


# 3. 사용자 인증 콜백 (로그인 시 유저별 대화방 목록 분리 관리)
@cl.password_auth_callback
def auth_callback(username: str, password: str) -> Optional[cl.User]:
    # 개발 및 실습용 기본 계정
    if (username == "user" and password == "1234") or (username == "admin" and password == "admin"):
        return cl.User(identifier=username, metadata={"role": username})
    return None

# 4. 에이전트 선택 프로필 (app/agents/ 디렉터리 동적 스캔)
@cl.set_chat_profiles
async def chat_profile(current_user: Optional[cl.User] = None):
    agents_dir = os.path.join(project_root, "app", "agents")
    profiles = []
    
    if not os.path.exists(agents_dir):
        return [
            cl.ChatProfile(
                name="chatbot",
                markdown_description="🤖 **Standard Chatbot**: 기본 대화 에이전트",
                icon="https://api.dicebear.com/7.x/bottts/svg?seed=chatbot"
            )
        ]
        
    for filename in sorted(os.listdir(agents_dir)):
        if filename.endswith(".py") and not filename.startswith("__") and filename != "utils.py":
            agent_name = filename[:-3]
            try:
                module_path = f"app.agents.{agent_name}"
                if module_path in sys.modules:
                    module = sys.modules[module_path]
                else:
                    module = importlib.import_module(module_path)
                
                meta = getattr(module, "AGENT_METADATA", {})
                name = meta.get("name", agent_name)
                desc = meta.get("description", f"🤖 **{agent_name.capitalize()}**: 런타임 로드 에이전트")
                icon = meta.get("icon", f"https://api.dicebear.com/7.x/bottts/svg?seed={agent_name}")
                
                profiles.append(
                    cl.ChatProfile(
                        name=name,
                        markdown_description=desc,
                        icon=icon
                    )
                )
            except Exception as scan_err:
                print(f"⚠️ Failed to parse metadata for {agent_name}: {scan_err}")
                profiles.append(
                    cl.ChatProfile(
                        name=agent_name,
                        markdown_description=f"🤖 **{agent_name.capitalize()}** Agent",
                        icon=f"https://api.dicebear.com/7.x/bottts/svg?seed={agent_name}"
                    )
                )
                
    return profiles or [
        cl.ChatProfile(
            name="chatbot",
            markdown_description="🤖 **Standard Chatbot**: 기본 대화 에이전트",
            icon="https://api.dicebear.com/7.x/bottts/svg?seed=chatbot"
        )
    ]

@cl.on_chat_start
async def on_chat_start():
    """새 대화방(New Chat) 시작 시 선택된 에이전트 동적 로드 및 상태 초기화"""
    profile = cl.user_session.get("chat_profile") or "chatbot"
    
    # 선택된 프로필에 맞는 에이전트 동적 로드
    agent_executor = await _get_agent_executor(profile)
    cl.user_session.set("agent_executor", agent_executor)
    
    # 안내 메시지 발송
    await cl.Message(
        content=f"👋 **안녕하세요! [{profile.upper()}] 와의 대화에 오신 것을 환영합니다.**\n과거 대화 목록은 왼쪽 사이드바에서 언제든지 확인하고 다시 열 수 있습니다.",
        author="Agent Assistant"
    ).send()

@cl.on_chat_resume
async def on_chat_resume(thread: ThreadDict):
    """왼쪽 사이드바에서 기존 대화방 클릭 시 해당 에이전트 세션 복원"""
    tags = thread.get("tags") or []
    profile = thread.get("metadata", {}).get("chat_profile") or (tags[0] if tags else "chatbot")
    
    agent_executor = await _get_agent_executor(profile)
    cl.user_session.set("agent_executor", agent_executor)
    cl.user_session.set("chat_profile", profile)
    
    await cl.Message(
        content=f"📂 **이전 대화방('{thread.get('name', 'Chat')}')이 성공적으로 복원되었습니다.**\n(에이전트: `{profile}`)\n이어서 지시사항을 입력해 주세요.",
        author="Agent Assistant"
    ).send()

@cl.on_message
async def on_message(message: cl.Message):
    """사용자 메시지 수신 및 LangGraph 에이전트 실시간 스트리밍 실행"""
    agent_executor = cl.user_session.get("agent_executor")
    
    # Chainlit이 관리하는 현재 대화방 ID (thread_id)
    thread_id = cl.context.session.thread_id or str(uuid.uuid4())
    
    if not agent_executor:
        profile = cl.user_session.get("chat_profile") or "chatbot"
        agent_executor = await _get_agent_executor(profile)
        cl.user_session.set("agent_executor", agent_executor)
    
    config = {
        "configurable": {"thread_id": thread_id},
        "recursion_limit": 100
    }
    
    context_obj = AgentContext(
        session_id=thread_id,
        logging_enabled=False,
        hitl_enabled=False,
    )
    
    final_message = cl.Message(content="")
    active_steps: Dict[str, cl.Step] = {}
    
    try:
        async for event in agent_executor.astream_events(
            {"messages": [("user", message.content)]},
            config=config,
            context=context_obj,
            version="v2"
        ):
            kind = event["event"]
            run_id = event.get("run_id", "")
            
            # 1. 도구 호출 시작 (Nested Step 트리 렌더링)
            if kind == "on_tool_start":
                tool_name = event.get("name", "Tool")
                tool_input = event.get("data", {}).get("input", "")
                
                step = cl.Step(name=f"🛠️ {tool_name}", type="tool")
                step.input = sanitize_text(str(tool_input))
                await step.send()
                active_steps[run_id] = step
                
            # 2. 도구 호출 완료
            elif kind == "on_tool_end":
                if run_id in active_steps:
                    step = active_steps[run_id]
                    tool_output = str(event.get("data", {}).get("output", ""))
                    truncated = tool_output[:1000] + "\n...(일부 생략)" if len(tool_output) > 1000 else tool_output
                    step.output = sanitize_text(truncated)
                    await step.update()
                    del active_steps[run_id]
                    
            # 3. 모델 토큰 스트리밍
            elif kind == "on_chat_model_stream":
                tags = event.get("tags", [])
                if "exclude_from_stream" in tags:
                    continue
                
                chunk = event.get("data", {}).get("chunk")
                if chunk and chunk.content:
                    normalized = sanitize_text(normalize_content(chunk.content))
                    if normalized:
                        await final_message.stream_token(normalized)
                        
        # 4. 이미지 태그 파싱 (<Render_Image>...</Render_Image>)
        raw_content = final_message.content or ""
        elements = []
        image_matches = re.findall(r"<Render_Image>(.*?)</Render_Image>", raw_content)
        for img_path in image_matches:
            img_path = img_path.strip()
            if os.path.exists(img_path):
                elements.append(cl.Image(name=os.path.basename(img_path), path=img_path, display="inline"))
                
        if elements:
            final_message.elements = elements
            
        await final_message.send()
        
    except Exception as e:
        await cl.Message(content=f"❌ **에러가 발생했습니다:** {str(e)}").send()
