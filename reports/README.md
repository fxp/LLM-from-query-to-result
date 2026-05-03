# Reports

Cold-start verification logs from rented GPU instances.
Each pair of files documents one independent end-to-end run from
empty server → web-served agent answering "1234+5678 → 6912":

| Date | Hardware | Region | Report | Raw log |
|---|---|---|---|---|
| 2026-05-03 | RTX 5090 (sm_120) | autodl 西区 D | [cold-start-2026-05-03-server3.md](cold-start-2026-05-03-server3.md) | [.raw.log](cold-start-2026-05-03-server3.raw.log) |

## What's in here

- `EXPERIMENT_REPORT.md` — 正式实验报告（约 6500 字）。HTML 版本：[`web/report.html`](../web/report.html)
- `*-report.md` — structured cold-start log with timing, observations, blog-ready commentary
- `*-raw.log` — exact stdout/stderr captured during the run, useful for debugging or re-derived numbers

These are a starting point for the project's blog/report writing — every
number cited in a post can trace back to an actual recorded run here.
