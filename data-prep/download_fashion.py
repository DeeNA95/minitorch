import os
import urllib.request
import gzip
import shutil

URLS = [
    'http://fashion-mnist.s3-website.eu-central-1.amazonaws.com/train-images-idx3-ubyte.gz',
    'http://fashion-mnist.s3-website.eu-central-1.amazonaws.com/train-labels-idx1-ubyte.gz',
    'http://fashion-mnist.s3-website.eu-central-1.amazonaws.com/t10k-images-idx3-ubyte.gz',
    'http://fashion-mnist.s3-website.eu-central-1.amazonaws.com/t10k-labels-idx1-ubyte.gz'
]

DATA_DIR = os.path.join(os.path.dirname(__file__), '../data/fashion_mnist')

if not os.path.exists(DATA_DIR):
    os.makedirs(DATA_DIR)

for url in URLS:
    filename = url.split('/')[-1]
    filepath = os.path.join(DATA_DIR, filename)
    
    print(f"Downloading {filename}...")
    urllib.request.urlretrieve(url, filepath)
    
    with gzip.open(filepath, 'rb') as f_in:
        unzipped_filepath = filepath.replace('.gz', '')
        print(f"Unzipping to {os.path.basename(unzipped_filepath)}...")
        with open(unzipped_filepath, 'wb') as f_out:
            shutil.copyfileobj(f_in, f_out)
            
    os.remove(filepath)

print("Fashion MNIST download complete. Ready for C++ indexing!")
