import random

# 카테고리별 점심 메뉴 목록
MENUS = {
    "한식": ["김치찌개", "된장찌개", "비빔밥", "제육볶음", "순두부찌개", "불고기", "국밥"],
    "중식": ["짜장면", "짬뽕", "볶음밥", "마파두부", "탕수육"],
    "일식": ["돈카츠", "초밥", "라멘", "규동", "우동", "가츠동"],
    "양식": ["파스타", "피자", "수제버거", "샐러드", "스테이크"],
    "분식/간편식": ["떡볶이", "김밥/라면", "샌드위치", "포케", "토스트"]
}

def recommend_lunch(category=None):
    """지정된 카테고리 또는 전체 메뉴 중 랜덤으로 추천"""
    if category and category in MENUS:
        selected_category = category
        menu = random.choice(MENUS[category])
    else:
        selected_category = random.choice(list(MENUS.keys()))
        menu = random.choice(MENUS[selected_category])
    
    return selected_category, menu

def main():
    print("=" * 40)
    print("🍽️  오늘의 점심 메뉴 추천기 🍽️")
    print("=" * 40)
    
    category, menu = recommend_lunch()
    print(f"\n오늘 추천하는 메뉴는 [{category}]의 ⭐ {menu} ⭐ 입니다!")
    print("맛있는 점심 식사 되세요! 😋\n")

if __name__ == "__main__":
    main()
