import os
import requests
import asyncio
import uuid
from typing import List
from dotenv import load_dotenv
from playwright.async_api import async_playwright

from google.cloud import storage
from google.adk.tools.tool_context import ToolContext

from .type import ImageURLs

load_dotenv()


async def search_google_lens_by_url(image_url: str, tool_context: ToolContext):
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
        result = []
        thumbnail_elements = await page.locator('div.kb0PBd img').all()
        
        # 7. 썸네일 이미지 링크 추출
        link_elements = await page.locator('a:has(span.Yt787)').all()

        result: List[ImageURLs] = []
        for index, (img, link) in enumerate(zip(thumbnail_elements[:5], link_elements[:5]), start=1):
            image_links = {}

            src = await img.get_attribute("src")
            if src and src.startswith("data:image/jpeg;base64,"):
                image_links[f'thumbnail_{index}'] = src

            href = await link.get_attribute("href")
            if href and href.startswith("http"):
                image_links[f'link_{index}'] = href

            result.append(image_links)

        print("\n🖼️")
        print(result)         

        await page.wait_for_timeout(2000)
        await browser.close()
        
        tool_context.state['image_links'] = result
        
        return result


def upload_image_to_gcs(
    tool_context: ToolContext,
    destination_blob_name: str,
    service_account_path: str = "service_account.json"
): 
    try:
        image_links: List[ImageURLs] = tool_context.state.get('image_links')
        
        for links, index in enumerate(image_links):
            # 1. 이미지 다운로드
            response = requests.get(links.image_url, stream=True, timeout=10)
            if response.status_code != 200:
                print("❌ 이미지 다운로드 실패:", response.status_code)
                return None

            # 2. 임시 파일로 저장
            temp_file = f"scraped/{uuid.uuid4()}/temp_image_{index}.png"
            with open(temp_file, "wb") as f:
                for chunk in response.iter_content(1024):
                    f.write(chunk)
            
            # ✅ 메타데이터 설정
            blob.metadata = {
                "source_url": links.website_link
            }

            # 3. GCS 클라이언트 설정
            storage_client = storage.Client.from_service_account_json(service_account_path)
            bucket = storage_client.bucket(os.getenv("GOOGLE_CLOUD_STORAGE_BUCKET"))
            blob = bucket.blob(destination_blob_name)
        
            # 4. 업로드
            blob.upload_from_filename(temp_file, content_type="image/png")
            blob.make_public()  # 퍼블릭 읽기 허용

            # 5. 임시 파일 삭제
            os.remove(temp_file)

            print("✅ 업로드 성공:", blob.public_url)
            print("   🔗 출처:", links.website_link)
            return blob.public_url

    except Exception as e:
        print("❌ 업로드 실패:", e)
        return None
    
    
# 실행 예시
# if __name__ == "__main__":
#     image_url = "https://storage.googleapis.com/artnest-suspected-images/artworks/jellyfish.png"
#     asyncio.run(search_google_lens_by_url(image_url))


