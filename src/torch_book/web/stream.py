from pathlib import Path
import httpx


def download(url: str, save_dir: str):
    """下载 url 到 save_dir"""
    save_dir = Path(save_dir)
    save_dir.mkdir(parents=True, exist_ok=True)
    headers = {
        'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/58.0.3029.110 Safari/537.3'
    }
    save_path = save_dir/url.split("/")[-1]
    try:
        with httpx.stream('GET', url, headers=headers) as response:
            response.raise_for_status()
            with open(save_path, 'wb') as f:
                for chunk in response.iter_bytes():
                    f.write(chunk)
    except httpx.HTTPError as e:
        print(f"An HTTP error occurred: {e}")
    except Exception as e:
        print(f"An error occurred: {e}")
    return save_path
