# GitHub Upload Guide

Recommended repository name:

```text
machilens-shibuya-ai-predictor
```

## Push from terminal

```bash
git init
git branch -M main
git add .
git commit -m "Initial release: MachiLens Shibuya AI demo"
git remote add origin https://github.com/YOUR_USERNAME/machilens-shibuya-ai-predictor.git
git push -u origin main
```

## Turn on GitHub Pages

```text
Settings → Pages → Deploy from a branch → main → /root → Save
```

The live demo will usually appear at:

```text
https://YOUR_USERNAME.github.io/machilens-shibuya-ai-predictor/
```

## What to show in the project summary

- `index.html` for the browser demo
- `README.md` for project explanation
- `src/machilens_ai/` for the AI pipeline
- `reports/model_card.md` and `reports/data_card.md` for limitations and data notes
