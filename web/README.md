# Web Pages

两个独立的、自包含的 HTML 页面：

- **`index.html`** — 项目落地页（hero + demo + 8 层架构 + 关键数字）
- **`report.html`** — 完整实验报告（学术风排版，TOC + 表格 + callouts）

两个文件都是单文件 inline CSS、自动 light/dark mode、移动端适配，不需要任何 build 步骤。

## 部署管线（GitHub Pages + Actions）

`.github/workflows/docs.yml` 把这两个 polished 页面和 mkdocs 站合并到一个 GitHub Pages：

| 路径 | 来源 |
|---|---|
| `/` | `web/index.html`（落地页） |
| `/report.html` | `web/report.html`（实验报告） |
| `/00-train/`, `/00b-sft/`, ..., `/05-gpu/`, `/trace/` | `docs/*.md`（mkdocs Material） |
| `/blog/`, `/blog/00-overview/`, ..., `/blog/10-L0.6-agent/`, `/blog/article/` | `blog/*.md`（mkdocs build 时 stage 进 `docs/blog/`） |

触发条件：`main` 分支上 `web/`、`blog/`、`docs/`、`mkdocs.yml`、`reports/EXPERIMENT_REPORT.md`、workflow 自身任一变化都会重建并推送到 `gh-pages` 分支。

要本地预览整套站点：

```bash
# stage blog 进 docs/（workflow 在 CI 上自动做这一步）
mkdir -p docs/blog && cp -r blog/* docs/blog/ && mv docs/blog/README.md docs/blog/index.md
mkdocs serve
# 另起一个 server 来看 web/*.html
python -m http.server -d web 8001
```

`index.html` 是一个独立的、自包含的 HTML 落地页，用来给项目做"门面"。

- 单文件，没有外部 JS/CSS 依赖
- inline CSS，支持 light/dark mode（跟系统主题）
- 响应式，移动端友好
- 引用项目内的 blog/article.md、reports/EXPERIMENT_REPORT.md 等

## 怎么部署

### 方式 1：直接打开本地文件
```bash
open web/index.html
```
（macOS / Linux 用 `xdg-open`）

### 方式 2：起一个 HTTP server
```bash
cd web && python -m http.server 8080
# 浏览器打开 http://localhost:8080
```

### 方式 3：GitHub Pages
1. Settings → Pages → Source: `main` branch, `/web` folder
2. 几分钟后访问 `https://<username>.github.io/<repo>/`

页面里的相对链接（`../blog/article.md` 等）在 GitHub Pages 上会自动 resolve 到 repo 文件——GitHub 渲染 .md 文件本身就有 markdown 支持。

### 方式 4：自己服务器
直接拷 `web/index.html` 到你的 nginx / static host 即可。

## 设计要点

- Hero 用 accent 色渐变背景突出标题
- 整体 typography 接近 [Anthropic 官网](https://anthropic.com) 的低调技术感
- 关键 demo 用 terminal-style 黑底框（💭 / 🔧 / ↳ / 🤖 emoji 区分 trace 步骤）
- 8 层架构用 grid 卡片，左 4px 边竖线
- 关键数字用大号绿字 metric cards
- 表格 + callout 强调最重要的论点（"1234+5678 → 6912"）

CSS variable 用 `--accent` 等，方便后续主题色调整。
