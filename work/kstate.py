"""Existence check for our own kernels that does not lie.

Two traps cost us time today (A201):
  * `kaggle kernels status <slug>` answers **404** for a kernel that DOES exist but was created
    moments ago -- so a lost push receipt plus a 404 looks exactly like "the push never happened",
    and a retry loop will keep re-pushing an existing kernel (each retry is a new version + a new
    GPU run);
  * `kaggle kernels list --user <u>` returns a truncated, oddly ordered view that hides private
    notebook titles.

`ListKernels(user=...)` is the reliable one.  This module caches one page-through so callers can
ask about many slugs cheaply.

Usage:  python work/kstate.py [substring]
"""
from __future__ import annotations

import sys

import requests

from kaggle.api.kaggle_api_extended import KaggleApi
from kagglesdk.kernels.types.kernels_api_service import ApiListKernelsRequest, KernelsListSortType

_CACHE: dict[str, dict] | None = None


def mine() -> dict[str, dict]:
    """slug -> kernel dict, for every kernel we own."""
    global _CACHE
    if _CACHE is not None:
        return _CACHE
    api = KaggleApi()
    api.authenticate()
    hc = api.build_kaggle_client()._http_client
    hc._init_session()
    url = hc._get_request_url('kernels.KernelsApiService', 'ListKernels')
    out: dict[str, dict] = {}
    token = None
    for _ in range(30):
        req = ApiListKernelsRequest()
        req.user = 'ppxl16'
        req.sort_by = KernelsListSortType.DATE_RUN
        req.page_size = 100
        if token:
            req.page_token = token
        hr = requests.Request(method='POST', url=url, json=req.__class__.to_dict(req),
                              headers=hc._session.headers, auth=hc._session.auth)
        r = hc._session.send(hr.prepare(), timeout=120)
        if r.status_code != 200:
            raise RuntimeError('ListKernels %s %s' % (r.status_code, r.text[:200]))
        d = r.json()
        ks = d.get('kernels', d.get('kernelsList', []))
        for k in ks:
            slug = str(k.get('ref') or k.get('scriptUrl') or '').split('/')[-1]
            if slug:
                out[slug] = k
        token = d.get('nextPageToken')
        if not token or not ks:
            break
    _CACHE = out
    return out


def forget() -> None:
    """Drop the cache so the next state() re-reads the server (a push may have just happened)."""
    global _CACHE
    _CACHE = None


def state(slug: str) -> str:
    """'EXISTS' (with its last run time) or 'ABSENT'."""
    k = mine().get(slug.split('/')[-1])
    if not k:
        return 'ABSENT'
    return 'EXISTS run=%s' % str(k.get('lastRunTime'))[:19]


if __name__ == '__main__':
    needle = (sys.argv[1] if len(sys.argv) > 1 else '').lower()
    ks = mine()
    print('kernels owned:', len(ks))
    for slug, k in sorted(ks.items(), key=lambda kv: str(kv[1].get('lastRunTime') or ''), reverse=True):
        if needle and needle not in slug.lower():
            continue
        print('%-46s run=%-20s score=%-7s' % (slug[:46], str(k.get('lastRunTime'))[:19], k.get('bestPublicScore')))
