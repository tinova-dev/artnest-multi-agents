import asyncio
from playwright.async_api import async_playwright

async def search_google_lens_by_url(image_url: str):
    async with async_playwright() as p:
        browser = await p.chromium.launch(headless=False)
        context = await browser.new_context(
            locale="en-US",
            extra_http_headers={"Accept-Language": "en-US,en;q=0.9"},
            timezone_id="America/Los_Angeles"
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

        # 5. 결과 스크린샷 저장
        await page.screenshot(path="lens_result.png")
        print("✅ Search complete. Result URL:", page.url)

        await browser.close()

# 실행 예시
if __name__ == "__main__":
    image_url = "https://storage.googleapis.com/artnest-suspected-images/artworks/jellyfish.png"
    asyncio.run(search_google_lens_by_url(image_url))
