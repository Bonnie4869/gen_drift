"""Download CSV source images without loading generation/evaluation models."""
import argparse
import csv
from pathlib import Path
import time
import requests

ROOT = Path(__file__).resolve().parent


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--start', type=int, default=0)
    p.add_argument('--stop', type=int, default=None, help='exclusive CSV index')
    p.add_argument('--retries', type=int, default=3)
    args = p.parse_args()
    with (ROOT / 'all_five_objects_sorted_by_attributes.csv').open(encoding='utf-8-sig', newline='') as f:
        rows = list(csv.DictReader(f))
    stop = len(rows) if args.stop is None else args.stop
    if not 0 <= args.start < stop <= len(rows) or args.retries < 1:
        p.error('invalid range or retries')
    output = ROOT / 'data/images'
    output.mkdir(parents=True, exist_ok=True)
    failures = []
    with requests.Session() as session:
        session.headers['User-Agent'] = 'Mozilla/5.0 GenDrift'
        for index in range(args.start, stop):
            row = rows[index]
            dest = output / f"{row['image_id']}.jpg"
            if dest.exists() and dest.stat().st_size:
                print(f'[{index+1}/{stop}] skip {dest.name}', flush=True)
                continue
            partial = dest.with_suffix('.jpg.part')
            for attempt in range(args.retries):
                try:
                    with session.get(row['url'], timeout=(20, 60), stream=True) as response:
                        response.raise_for_status()
                        with partial.open('wb') as handle:
                            for chunk in response.iter_content(1024 * 256):
                                handle.write(chunk)
                    if not partial.stat().st_size:
                        raise ValueError('empty image response')
                    from PIL import Image
                    with Image.open(partial) as image:
                        image.verify()
                    partial.replace(dest)
                    print(f'[{index+1}/{stop}] saved {dest.name}', flush=True)
                    break
                except (requests.RequestException, OSError, ValueError) as error:
                    print(f'[{index+1}/{stop}] attempt {attempt+1}: {error}', flush=True)
                    partial.unlink(missing_ok=True)
                    if attempt + 1 == args.retries:
                        failures.append({'index': index, 'image_id': row['image_id'], 'url': row['url'], 'error': str(error)})
                    else:
                        time.sleep(2)
    report = output / f'download_failures_{args.start}_{stop}.csv'
    with report.open('w', newline='', encoding='utf-8-sig') as f:
        writer = csv.DictWriter(f, fieldnames=['index', 'image_id', 'url', 'error'])
        writer.writeheader()
        writer.writerows(failures)
    print(f'Finished: {len(failures)} failures. Report: {report}', flush=True)
    if failures:
        raise SystemExit(1)


if __name__ == '__main__':
    main()
