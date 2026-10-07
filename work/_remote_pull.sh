#!/bin/bash
# Corrected corpus pull: the downloader takes --url <tsv> and --outdir <dir>, not --dst.
set -u
cd /root
for pair in "urls_corpus.tsv:knee-raptor-corpus" "urls_corpus_ext.tsv:knee-raptor-corpus-ext"; do
  tsv="${pair%%:*}"; slug="${pair##*:}"
  dst="/kaggle/input/datasets/dreaddevelopment/$slug"
  mkdir -p "$dst"
  nohup /root/miniconda3/bin/python /root/kaggle_parallel.py --url "/root/$tsv" --outdir "$dst" --streams 12 \
        > "/root/pull_$slug.log" 2>&1 &
  echo "started $slug -> $dst (pid $!)"
done
sleep 45
echo "=== progress after 45s ==="
du -sh /kaggle/input/datasets/dreaddevelopment/* 2>/dev/null
echo "--- corpus log tail ---"; tail -4 /root/pull_knee-raptor-corpus.log 2>&1
echo "--- ext log tail ---";    tail -4 /root/pull_knee-raptor-corpus-ext.log 2>&1
