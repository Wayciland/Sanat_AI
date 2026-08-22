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
    print("=== ENIMERA LENS: DİJİTAL İNSAN SANATI ODAKLI TOPLAYICI ===")
    
    # 1. Yapay Zeka Sanatı (Yapay Zeka Dijital Çizimleri)
    indir_bing("midjourney v6 digital art portrait", "dataset/ai", limit=120)
    indir_bing("stable diffusion xl anime digital illustration", "dataset/ai", limit=120)
    
    # 2. İnsan Sanatı (Ağırlıklı Dijital Tablet ve Lineart Çizimleri)
    indir_bing("digital 2d portrait painting human artist", "dataset/insan", limit=120)
    indir_bing("clip studio paint digital illustration process", "dataset/insan", limit=120)
    indir_bing("digital character concept art speedpaint human", "dataset/insan", limit=120)
    indir_bing("traditional oil painting museum scan", "dataset/insan", limit=120)
    
    print("\n[BAŞARILI] Dijital insan sanati ağirlikli veri seti indirildi!")