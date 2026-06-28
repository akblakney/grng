just starting


- sudo apt install portaudio19-dev

# Workflow for `grng-collect`

- Run grng-collect: `grng-collect --output-dir /base/dir --sample-rate 48000 --validate 0.005 audio`
    - this will put all raw generated data in `/base/dir/raw`
- Run grng-collect-post: `grng-collect-post /base/dir`
    - generate `daily_stats.json` in each date directory in `/base/dir/raw/`
    - generate `daily.bin` in each date directory `/base/dir/post` which holds the post-processed daily raw data
    - generate `daily_stats.json` and `distill_report.json` alongside `daily.bin` which hold stats and report
- Optionally, see summarized statistics across all dates for both `raw` and `post` data  with
    - `grng-collect-summarize-stats /base/dir <test>`
    - e.g. `grng-collect-summarize-stats /base/dir monobit`
