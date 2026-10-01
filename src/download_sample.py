import requests
import os

def download_partial_csv():
    url = "https://huggingface.co/datasets/c01dsnap/CIC-IDS2017/resolve/main/Friday-WorkingHours-Afternoon-PortScan.pcap_ISCX.csv"
    print(f"Streaming first few megabytes from {url}...")
    
    os.makedirs("data/raw", exist_ok=True)
    dest_path = "data/raw/Friday_PortScan_sample.csv"
    
    try:
        response = requests.get(url, stream=True, timeout=10)
        response.raise_for_status()
        
        with open(dest_path, "wb") as f:
            downloaded = 0
            # Download just 5MB to get enough rows for a real EDA
            target_size = 5 * 1024 * 1024 
            for chunk in response.iter_content(chunk_size=8192):
                if chunk:
                    f.write(chunk)
                    downloaded += len(chunk)
                    if downloaded >= target_size:
                        break
        
        # Now clean up the last partial line
        with open(dest_path, "r", encoding="utf-8", errors="ignore") as f:
            lines = f.readlines()
            
        with open(dest_path, "w", encoding="utf-8") as f:
            # write all but the last line (which is likely incomplete)
            f.writelines(lines[:-1])
            
        print(f"Successfully downloaded partial dataset ({len(lines)-1} rows) to {dest_path}")
    except Exception as e:
        print(f"Error: {e}")

if __name__ == "__main__":
    download_partial_csv()
