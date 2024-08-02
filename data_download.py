import requests

record_id = "8034596"
r = requests.get(f"https://zenodo.org/api/records/{record_id}")
download_urls = [f["links"]["self"] for f in r.json()["files"]]
filenames = [f["key"] for f in r.json()["files"]]

for filename, url in zip(filenames, download_urls):
    print("Downloading:", filename)
    r = requests.get(url)
    with open(filename, "wb") as f:
        f.write(r.content)
