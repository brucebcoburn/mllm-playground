import os
import gdown

# This establishes the top-level 'mllm-playground' directory
BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
TEST_IMAGES_DIR = os.path.join(BASE_DIR, "test_images")

VALID_EXTS = (".png", ".jpg", ".jpeg", ".bmp", ".webp")


def download_test_data(target_path=TEST_IMAGES_DIR):
    """
    Downloads our test_images/ directory from Google Drive
    NOTE: the folder url is currently hardcoded in...
    """
    folder_url = "https://drive.google.com/drive/folders/13680SqnUnhrRmAuNxz_CU8xB6MbjyXUO?usp=sharing"

    if not os.path.exists(target_path) or len(os.listdir(target_path)) <= 1:
        print(f"Downloading test data from Google Drive to {target_path}...")
        try:
            os.makedirs(target_path, exist_ok=True)
            gdown.download_folder(
                url=folder_url, output=target_path, quiet=False, use_cookies=False
            )
        except Exception as e:
            print(
                f"Error downloading folder: {e}. Ensure 'pip install gdown' is installed."
            )
    else:
        print(f"Test images are ready at {target_path}")
