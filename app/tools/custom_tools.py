"""
app/tools/custom_tools.py

🎯 Mission 01: 커스텀 도구(Custom Tool) 기준 완성본
주사위 굴리기(roll_dice)와 환율 계산(convert_currency) 도구가 구현되어 있습니다.
"""

import random
from langchain_core.tools import tool


@tool(parse_docstring=True)
def roll_dice(num_dice: int = 1, sides: int = 6) -> str:
    """지정된 개수와 면을 가진 주사위를 굴려 무작위 결과를 반환합니다.
    
    사용자가 주사위 굴리기, 게임 승패 결정, 난수 생성, 무작위 번호 뽑기를 요청할 때 반드시 호출하세요.

    Args:
        num_dice: 굴릴 주사위 개수 (기본값: 1, 최대: 10)
        sides: 주사위의 면 수 (기본값: 6, 예: 6, 12, 20)
    """
    num_dice = max(1, min(num_dice, 10))
    rolls = [random.randint(1, sides) for _ in range(num_dice)]
    total = sum(rolls)
    return f"🎲 주사위 {num_dice}개(d{sides}) 결과: {rolls} (합계: {total})"


@tool(parse_docstring=True)
def convert_currency(amount: float, from_currency: str = "USD", to_currency: str = "KRW") -> str:
    """주요 국가 통화 간의 환율을 계산하여 환전 금액을 반환합니다.
    
    사용자가 달러, 유로, 엔화, 원화 등의 환율 조회나 환전 계산을 요청할 때 반드시 호출하세요.

    Args:
        amount: 환전할 금액
        from_currency: 기준 통화 코드 (USD, EUR, JPY, KRW)
        to_currency: 대상 통화 코드 (기본값: KRW)
    """
    # 모의 고정 환율 데이터 (실전에서는 외부 환율 API 연동 가능)
    rates_to_krw = {
        "USD": 1380.0,
        "EUR": 1500.0,
        "JPY": 9.2,    # 1엔당 원화
        "KRW": 1.0
    }
    
    from_curr = from_currency.upper()
    to_curr = to_currency.upper()
    
    if from_curr not in rates_to_krw or to_curr not in rates_to_krw:
        return f"⚠️ 지원하지 않는 통화입니다. 지원 목록: {list(rates_to_krw.keys())}"
        
    in_krw = amount * rates_to_krw[from_curr]
    result = in_krw / rates_to_krw[to_curr]
    
    return f"💱 환율 계산: {amount:,.2f} {from_curr} = {result:,.2f} {to_curr} (기준환율: 1 {from_curr}당 {rates_to_krw[from_curr]}원)"
