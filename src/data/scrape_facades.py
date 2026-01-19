"""
Web scraping pipeline for collecting unlabeled facade images.

This script focuses on Dutch housing/facades from Wikimedia Commons.
Uses the Wikimedia API to search for and download images.

Usage:
    python -m src.data.scrape_facades --limit 10 --output data/unlabeled
    python -m src.data.scrape_facades --limit 500 --output data/unlabeled --query "Amsterdam houses"
"""

import os
import time
import json
import hashlib
import argparse
import requests
from pathlib import Path
from concurrent.futures import ThreadPoolExecutor, as_completed
from typing import Optional
from urllib.parse import quote

# Wikimedia Commons API endpoint
WIKI_API = "https://commons.wikimedia.org/w/api.php"

# Search queries focused on Dutch facades and European architecture
DUTCH_FACADE_QUERIES = [
    "Dutch house facade",
    "Amsterdam canal house", 
    "Netherlands row houses",
    "Dutch brick building",
    "Grachtenpand Amsterdam",
    "Rotterdam architecture",
    "Dutch townhouse",
    "Netherlands residential building",
    "Haarlem historic buildings",
    "Utrecht canal houses",
]


class FacadeScraper:
    """
    Downloads facade images from Wikimedia Commons.
    
    Wikimedia is chosen because:
    - Images are CC-licensed (legal for training)
    - Good quality architectural photos
    - Strong metadata
    """
    
    def __init__(self, output_dir: str, limit: int = 10):
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)
        self.limit = limit
        self.downloaded = 0
        self.session = requests.Session()
        self.session.headers.update({
            'User-Agent': 'FacadeDataCollector/1.0 (Research; https://github.com/example)'
        })
        
        # Load existing hashes to avoid duplicates
        self.hash_file = self.output_dir / "downloaded_hashes.json"
        self.downloaded_hashes = self._load_hashes()
        
    def _load_hashes(self) -> set:
        """Load previously downloaded image hashes."""
        if self.hash_file.exists():
            with open(self.hash_file, 'r') as f:
                return set(json.load(f))
        return set()
    
    def _save_hashes(self):
        """Save downloaded image hashes."""
        with open(self.hash_file, 'w') as f:
            json.dump(list(self.downloaded_hashes), f)
            
    def _get_image_hash(self, content: bytes) -> str:
        """Generate hash of image content for deduplication."""
        return hashlib.md5(content).hexdigest()
    
    def search_wikimedia(self, query: str, limit: int = 50) -> list:
        """
        Search Wikimedia Commons for images matching query.
        
        Returns list of image info dictionaries.
        """
        params = {
            'action': 'query',
            'generator': 'search',
            'gsrsearch': f'filetype:bitmap {query}',
            'gsrlimit': min(limit, 50),
            'gsrnamespace': 6,  # File namespace
            'prop': 'imageinfo',
            'iiprop': 'url|size|extmetadata',
            'format': 'json',
        }
        
        try:
            response = self.session.get(WIKI_API, params=params, timeout=30)
            response.raise_for_status()
            data = response.json()
            
            if 'query' not in data or 'pages' not in data['query']:
                return []
                
            results = []
            for page_id, page_data in data['query']['pages'].items():
                if 'imageinfo' in page_data:
                    info = page_data['imageinfo'][0]
                    
                    # Filter by size (want reasonably high-res images)
                    if info.get('width', 0) < 400 or info.get('height', 0) < 400:
                        continue
                        
                    # Filter by aspect ratio (facades are usually taller than wide or square-ish)
                    width = info.get('width', 1)
                    height = info.get('height', 1)
                    aspect = width / height
                    if aspect > 3 or aspect < 0.3:  # Skip panoramas and very thin images
                        continue
                    
                    results.append({
                        'title': page_data.get('title', ''),
                        'url': info.get('url'),
                        'width': width,
                        'height': height,
                    })
                    
            return results
            
        except Exception as e:
            print(f"Error searching Wikimedia: {e}")
            return []
    
    def download_image(self, image_info: dict) -> Optional[str]:
        """
        Download a single image.
        
        Returns the filepath if successful, None otherwise.
        """
        url = image_info.get('url')
        if not url:
            return None
            
        try:
            response = self.session.get(url, timeout=60)
            response.raise_for_status()
            content = response.content
            
            # Check for duplicates
            img_hash = self._get_image_hash(content)
            if img_hash in self.downloaded_hashes:
                return None
            
            # Determine filename from URL
            ext = url.split('.')[-1].lower()
            if ext not in ['jpg', 'jpeg', 'png', 'webp']:
                ext = 'jpg'
            
            filename = f"facade_{img_hash[:12]}.{ext}"
            filepath = self.output_dir / filename
            
            with open(filepath, 'wb') as f:
                f.write(content)
            
            self.downloaded_hashes.add(img_hash)
            return str(filepath)
            
        except Exception as e:
            print(f"Error downloading {url}: {e}")
            return None
    
    def scrape(self, queries: Optional[list] = None, progress_callback=None):
        """
        Main scraping loop.
        
        Args:
            queries: List of search queries. Uses DUTCH_FACADE_QUERIES if None.
            progress_callback: Optional callback(downloaded, total) for progress.
        """
        if queries is None:
            queries = DUTCH_FACADE_QUERIES
            
        all_images = []
        
        print(f"Searching for facade images (limit: {self.limit})...")
        
        # Collect image URLs from all queries
        for query in queries:
            if len(all_images) >= self.limit * 2:  # Collect extra for filtering
                break
                
            print(f"  Searching: '{query}'")
            results = self.search_wikimedia(query, limit=min(50, self.limit))
            all_images.extend(results)
            time.sleep(0.5)  # Rate limiting
        
        # Deduplicate by URL
        seen_urls = set()
        unique_images = []
        for img in all_images:
            if img['url'] not in seen_urls:
                seen_urls.add(img['url'])
                unique_images.append(img)
        
        print(f"Found {len(unique_images)} unique images, downloading up to {self.limit}...")
        
        # Download images
        downloaded_files = []
        
        for i, img_info in enumerate(unique_images):
            if self.downloaded >= self.limit:
                break
                
            filepath = self.download_image(img_info)
            if filepath:
                self.downloaded += 1
                downloaded_files.append(filepath)
                print(f"  [{self.downloaded}/{self.limit}] Downloaded: {os.path.basename(filepath)}")
                
                if progress_callback:
                    progress_callback(self.downloaded, self.limit)
            
            time.sleep(0.3)  # Rate limiting
        
        # Save hashes
        self._save_hashes()
        
        print(f"\nDownloaded {self.downloaded} images to {self.output_dir}")
        
        # Save metadata
        metadata_file = self.output_dir / "scrape_metadata.json"
        with open(metadata_file, 'w') as f:
            json.dump({
                'queries_used': queries,
                'total_downloaded': self.downloaded,
                'files': downloaded_files,
            }, f, indent=2)
        
        return downloaded_files


def main():
    parser = argparse.ArgumentParser(
        description="Scrape facade images from Wikimedia Commons"
    )
    parser.add_argument(
        '--output', '-o',
        type=str,
        default='data/unlabeled',
        help='Output directory for downloaded images'
    )
    parser.add_argument(
        '--limit', '-n',
        type=int,
        default=10,
        help='Maximum number of images to download'
    )
    parser.add_argument(
        '--query', '-q',
        type=str,
        default=None,
        help='Custom search query (uses default Dutch queries if not specified)'
    )
    
    args = parser.parse_args()
    
    queries = [args.query] if args.query else None
    
    scraper = FacadeScraper(
        output_dir=args.output,
        limit=args.limit
    )
    
    scraper.scrape(queries=queries)


if __name__ == "__main__":
    main()
