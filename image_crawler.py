import aiohttp
import asyncio
import aiofiles
from pathlib import Path
from typing import List, Set
import hashlib
from PIL import Image
import io
import numpy as np
import logging

logger = logging.getLogger(__name__)
logging.basicConfig(level=logging.INFO)


class ImageCrawler:
    def __init__(self, output_dir: str = "data/datasets", max_concurrent: int = 5):
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)

        self.seen_hashes: Set[str] = set()
        self.semaphore = asyncio.Semaphore(max_concurrent)

        # 🔒 ACTIVE CRAWL LOCK
        self.active_crawls: Set[str] = set()

        # ⚠️ MOVE API KEY TO ENV IN REAL PROJECT
        self.api_key = "jdzljmyMog16qUyeYuOeOnXgPT71kPLdqZ8cCxeNd4XXTmMBoSpH3iH0"
        if not self.api_key:
            raise RuntimeError("PEXELS_API_KEY not set")

    # =========================
    # PUBLIC API
    # =========================
    async def crawl_class(self, class_name: str, max_images: int = 100) -> int:
        if class_name in self.active_crawls:
            logger.warning(f"[{class_name}] Crawl already running")
            return 0

        self.active_crawls.add(class_name)
        logger.info(f"[{class_name}] Crawling started")

        try:
            class_dir = self.output_dir / class_name
            class_dir.mkdir(parents=True, exist_ok=True)

            urls = await self._pexels_image_urls(class_name, max_images)

            if not urls:
                logger.warning(f"[{class_name}] No images returned")
                return 0

            async with aiohttp.ClientSession() as session:
                tasks = [
                    self._bounded_download(session, url, class_dir)
                    for url in urls
                ]
                results = await asyncio.gather(*tasks, return_exceptions=True)

            downloaded = sum(r for r in results if isinstance(r, int))
            logger.info(f"[{class_name}] Downloaded {downloaded} images")
            return downloaded

        finally:
            self.active_crawls.discard(class_name)
            logger.info(f"[{class_name}] Crawling finished")

    # =========================
    # PEXELS SEARCH
    # =========================
    async def _pexels_image_urls(self, query: str, max_images: int) -> List[str]:
        urls: List[str] = []
        page = 1
        per_page = min(80, max_images)

        headers = {"Authorization": self.api_key}

        async with aiohttp.ClientSession(headers=headers) as session:
            while len(urls) < max_images:
                params = {
                    "query": query,
                    "per_page": per_page,
                    "page": page
                }

                async with session.get(
                    "https://api.pexels.com/v1/search",
                    params=params,
                    timeout=20
                ) as res:
                    if res.status != 200:
                        logger.warning(f"[{query}] Pexels HTTP {res.status}")
                        break

                    data = await res.json()

                photos = data.get("photos", [])
                if not photos:
                    break

                for p in photos:
                    src = p.get("src", {})
                    url = src.get("large") or src.get("original")
                    if url:
                        urls.append(url)
                    if len(urls) >= max_images:
                        break

                page += 1

        return urls

    # =========================
    # DOWNLOAD PIPELINE
    # =========================
    async def _bounded_download(self, session, url: str, output_dir: Path) -> int:
        async with self.semaphore:
            return await self._download_image(session, url, output_dir)

    async def _download_image(self, session, url: str, output_dir: Path) -> int:
        try:
            async with session.get(url, timeout=20) as res:
                if res.status != 200:
                    return 0
                image_data = await res.read()

            image_hash = hashlib.md5(image_data).hexdigest()
            if image_hash in self.seen_hashes:
                return 0

            if not self._validate_image(image_data):
                return 0

            async with aiofiles.open(output_dir / f"{image_hash}.jpg", "wb") as f:
                await f.write(image_data)

            self.seen_hashes.add(image_hash)
            return 1

        except Exception:
            return 0

    # =========================
    # IMAGE VALIDATION
    # =========================
    def _validate_image(self, image_data: bytes) -> bool:
        try:
            img = Image.open(io.BytesIO(image_data))
            img.verify()
            img = Image.open(io.BytesIO(image_data))

            if img.width < 224 or img.height < 224:
                return False

            arr = np.array(img)
            return arr.std() >= 10

        except Exception:
            return False
