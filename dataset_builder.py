import os
from icrawler.builtin import BingImageCrawler

def indir_bing(arama_terimi, klasor_adi, limit=120):
    os.makedirs(klasor_adi, exist_ok=True)
    print(f"\n==========================================")
    print(f"[{arama_terimi}] indiriliyor... (Hedef: {limit})")
    print(f"==========================================")
    
    bing_crawler = BingImageCrawler(
        downloader_threads=4,
        storage={'root_dir': klasor_adi}
    )
    bing_crawler.crawl(keyword=arama_terimi, max_num=limit)

if __name__ == "__main__":
    print("=== ENIMERA LENS: DİJİTAL VE GELENEKSEL VERİ TOPLAYICI ===")
    
    # 1. Yapay Zeka Sanatı
    indir_bing("midjourney v6 digital art", "dataset/ai", limit=120)
    indir_bing("stable diffusion xl portrait", "dataset/ai", limit=120)
    
    # 2. İnsan Sanatı (Geleneksel + Dijital Tablet)
    indir_bing("louvre museum oil painting scan", "dataset/insan", limit=120)
    indir_bing("classical impressionism canvas painting", "dataset/insan", limit=120)
    indir_bing("digital tablet concept art process human artist", "dataset/insan", limit=120) # Dijital İnsan Sanatı
    
    print("\n[BAŞARILI] Dijital insan sanati da veri setine başariyla eklendi!")