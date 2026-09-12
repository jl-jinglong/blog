
这是一个基于 [Chirpy](https://github.com/cotes2020/jekyll-theme-chirpy) 的中文研究博客，部署地址为：

`https://jl-jinglong.github.io/blog/`

## 写文章

在 `_posts/` 下新建 Markdown 文件，文件名使用：

`YYYY-MM-DD-title.md`

示例：

```markdown
---
title: "文章标题"
date: 2026-09-12 20:00:00 +0800
categories: [研究笔记]
tags: [AI, 机器学习]
---

正文写在这里。
```

## 本地预览

安装 Ruby 与 Bundler 后运行：

```bash
bundle install
bundle exec jekyll serve
```

然后访问 `http://127.0.0.1:4000/blog/`。

## 发布

将本目录推送到 GitHub 的 `jl-jinglong/blog` 仓库，并在仓库 Settings → Pages 中把发布来源设为 **GitHub Actions**。工作流位于 `.github/workflows/pages-deploy.yml`。

