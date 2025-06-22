import asyncio
from playwright.async_api import async_playwright

async def search_google_lens_by_url(image_url: str):
    async with async_playwright() as p:
        browser = await p.chromium.launch(headless=False, slow_mo=300)
        context = await browser.new_context(
            locale="en-US",
            extra_http_headers={"Accept-Language": "en-US,en;q=0.9"},
            timezone_id="America/Los_Angeles",
            user_agent="Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/123.0.0.0 Safari/537.36"
        )
        page = await context.new_page()

        # 1. Google Lens 접속
        await page.goto("https://www.google.com/webhp?hl=en", timeout=60000)
        await page.wait_for_timeout(2000)  # 렌더링 안정화
        
        # 2. "Search by image" 버튼 클릭 (div.nDcEnd)
        try:
            await page.locator('div.nDcEnd[role="button"]').click()
            await page.wait_for_timeout(1000)
        except Exception as e:
            print("⚠️ Search by image 버튼 클릭 실패:", e)

        try:
            # 2. input[type="url"] 필드 직접 탐색
            input_box = page.locator('input.cB9M7')
            await input_box.wait_for(timeout=5000)
            await input_box.fill(image_url)
        except Exception as e:
            print("⚠️ URL 입력창을 찾지 못했습니다:", e)
            await page.screenshot(path="error_before_input.png")
            await browser.close()
            return

        try:
            # 3. Search 버튼 클릭
            search_button = page.locator('div.Qwbd3[role="button"]')
            await search_button.wait_for(timeout=5000)
            await search_button.click()
        except Exception:
            # fallback: Enter로 대체
            await input_box.press("Enter")

        # 4. 결과 기다리기
        await page.wait_for_load_state("networkidle", timeout=10000)
        await page.wait_for_timeout(3000)  # 결과 로딩 보조 대기
        
        # 🖱️ 마우스/키보드 움직임 흉내내기
        await page.mouse.move(100, 100)
        await page.wait_for_timeout(500)
        await page.mouse.move(300, 200)
        await page.wait_for_timeout(500)
        await page.keyboard.press("Tab")
        await page.wait_for_timeout(300)
        await page.keyboard.type("Hello", delay=200)
        await page.wait_for_timeout(500)

        # 5. "Visual matches" 탭 클릭
        try:
            await page.locator('div.YmvwI', has_text="Visual matches").click()
            await page.wait_for_timeout(2000)
        except Exception as e:
            print("⚠️ Visual matches 탭 클릭 실패:", e)

        # 6. 썸네일 이미지 (Base64 or CDN) 추출
        thumbnail_elements = await page.locator("img").all()

        thumbnail_links = []
        for img in thumbnail_elements:
            src = await img.get_attribute("src")
            if src and src.startswith("data:image/jpeg;base64,"):
                thumbnail_links.append(src)

        thumbnail_links = thumbnail_links[:10]

        print("\n🖼️ Visual Matches 상위 10개 썸네일:")
        for link in thumbnail_links:
            print(" -", link)
            
        # 7. 썸네일 이미지 링크 추출
        link_elements = await page.locator('a:has(span.Yt787)').all()

        source_links = []
        for link in link_elements:
            href = await link.get_attribute("href")
            if href and href.startswith("http"):
                source_links.append(href)

        # 상위 10개만 출력
        source_links = source_links[:10]
        print("\n🌐 원본 웹사이트 링크:")
        for link in source_links:
            print(" -", link)

        await browser.close()

# 실행 예시
if __name__ == "__main__":
    image_url = "https://storage.googleapis.com/artnest-suspected-images/artworks/jellyfish.png"
    asyncio.run(search_google_lens_by_url(image_url))


