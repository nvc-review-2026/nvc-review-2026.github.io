# nvc-review2025.github.io

Minimal starter for publishing `https://nvc-review2025.github.io/` with GitHub Pages.

## Files

- `index.html`: page content
- `style.css`: simple styling

## Publish Steps

1. Commit and push to `main`:

```bash
git add .
git commit -m "Initialize minimal GitHub Pages site"
git push -u origin main
```

2. In GitHub repo settings, confirm:
   - `Settings` -> `Pages`
   - `Build and deployment` -> `Source: Deploy from a branch`
   - `Branch: main` and folder `/ (root)`

3. Wait about 1-3 minutes, then open:
   - `https://nvc-review2025.github.io/`

## Notes

- Because this repository is named exactly `nvc-review2025.github.io`, it is a user site.
- The site root URL maps directly to this repo's `main` branch root.
- You can safely iterate design/content later without changing deployment setup.
