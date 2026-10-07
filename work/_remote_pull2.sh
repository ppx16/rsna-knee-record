#!/bin/bash
# Pull each corpus FILE with its own pre-signed URL, N streams, and connection rotation.
#
# kaggle_parallel.py takes ONE --url and ONE --out, so the TSV is walked here. --rotate-sec is set because
# FACTS A43/A66 measured that a throttled connection can deliver ~0 MB/s forever without ever erroring, so a
# byte threshold alone never fires.
set -u
cd /root

pull_one () {
  local tsv="$1" dst="$2" tag="$3"
  mkdir -p "$dst"
  while IFS=$'\t' read -r name url; do
    [ -z "${name:-}" ] && continue
    echo "[$tag] $name"
    /root/miniconda3/bin/python /root/kaggle_parallel.py --url "$url" --out "$dst/$name" \
        --streams 12 --rotate-mb 32 --rotate-sec 12 >> "/root/pull_$tag.log" 2>&1
  done < "$tsv"
  echo "[$tag] DONE" >> "/root/pull_$tag.log"
}

pull_one /root/urls_corpus.tsv     /kaggle/input/datasets/dreaddevelopment/knee-raptor-corpus     corpus &
pull_one /root/urls_corpus_ext.tsv /kaggle/input/datasets/dreaddevelopment/knee-raptor-corpus-ext ext &

echo "launched both; waiting 60s to sample the rate"
sleep 60
du -sh /kaggle/input/datasets/dreaddevelopment/* 2>/dev/null
echo "--- corpus ---"; tail -3 /root/pull_corpus.log 2>&1
echo "--- ext ---";    tail -3 /root/pull_ext.log 2>&1
