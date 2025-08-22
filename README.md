# oss-benchmark
Benchmarking OSS model performance on-device for medical tasks.

### Setup
```
pip install cloudscraper beautifulsoup4
```

### Data
Get a single case
```
python data/get_case_eurorad.py https://www.eurorad.org/case/18706
```

Get all 2025 cases
```
python data/get_range_eurorad.py --start 18806 --end 19164 --outdir eurorad_csvs
```