# Qwen 3.8 team site

A self-contained static microsite. The published page is `index.html` with the bundled local traffic frame at `assets/videoframe_40129.png`; there is no build step or external asset dependency.

## Preview locally

From the project root:

```powershell
python -m http.server 8000 --directory team_site
```

Then open <http://localhost:8000>.

## Publish with GitHub Pages

The full project repository has a root workflow at `.github/workflows/pages.yml` that publishes only this `team_site` directory whenever `main` is updated. In the repository settings, select **Pages → Build and deployment → GitHub Actions**.

No build step or external asset dependency is required. A public URL still requires a target GitHub repository and authorized push access.
